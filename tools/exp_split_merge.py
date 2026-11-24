#!/usr/bin/env python3
"""EXPERIMENT: fragment handling variants vs the current clustering, offline.

Not production code: lidar/cluster_extractor.py, object_estimator.py and
object_tracker.py are not changed. Variants wrap the Euclidean extractor
(eval_objects_gt.EuclideanClusterExtractor is replaced while a variant runs)
or remap track IDs after tracking. LiDAR objects have no class; every rule
uses geometry only.

Variants:
    B0   Euclidean 0.6 m (current)
    T8   Euclidean 0.8 m (reference: the tolerance increase of the 1st
         report, which merged more GT objects)
    C1   B0 + contained-fragment merge: a cluster F joins cluster B when
           points(F) <= ratio * points(B)            (--ratio 0.3)
           >= contained of F's points lie in B's xy box widened by
           --widen (0.3 m)                            (--contained 0.8)
           min 3D point gap(F, B) <= --gap (1.2 m)
         and F is NOT a standing column: bottom < 0.3 m above ground,
         height >= 1.0 m and footprint <= 1.2 m (a person-sized object is
         never merged). One pass, no chains (a parent is never a fragment).
    C1b  C1 with ratio 0.5, contained 0.75
    C1h  C1b with containment in the convex hull of the body's xy points
         (+ widen) instead of its axis-aligned box: the empty corners of
         the box of a rotated vehicle do not count as inside
    C2   clusters unchanged (B0); a confirmed track whose object meets the
         C1b relation to the same other track for >= 3 consecutive frames
         is reported under that track's group ID (the older track), as an
         object-level link
Metrics (same definitions as the existing tools): GT detection (ok share),
xy / distance error, ID switches, fragments (eval_tracking_gt.summarize_object
on the reported ID), split metrics (eval_split_clusters: fragments, separate
confirmed fragment tracks = fragment reported ID != body reported ID, GT-GT
merges), Kalman velocity error of moving GT, extractor time per frame.

Safety (--inject): synthetic people (0.45 x 1.7 m, sensor-facing plane) and
cones (0.3 x 0.7 m) placed beside each GT vehicle at horizontal gaps
0.4 / 0.7 / 1.0 / 1.5 m (front and side) in sampled frames; result per
variant: separate (own cluster, >= 80 % of its points, purity >= 0.8),
absorbed (in a cluster made mostly of other points) or lost.

Usage:
    python3 tools/exp_split_merge.py <bag-dir>... [--variants B0,C1] \
        [--inject] [--json out.json]
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lidar"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_objects_gt  # noqa: E402
from cluster_extractor import ClusterParams, EuclideanClusterExtractor  # noqa
from eval_objects_gt import (  # noqa: E402
    EvalConfig, UnsupportedBag, bag_dirs, evaluate_frame, gt_boxes,
    load_preprocessed, summarize)
from eval_split_clusters import frame_split, summarize_gt  # noqa: E402
from eval_tracking_gt import summarize_object  # noqa: E402
from object_estimator import height_band_mask  # noqa: E402
from object_tracker import ObjectTracker, TrackerParams  # noqa: E402

GROUND_Z = 0.0
VEHICLE = ("car", "truck", "bus", "forklift", "vehicle")


def standing_column(world):
    lo, hi = world.min(axis=0), world.max(axis=0)
    return (lo[2] - GROUND_Z < 0.3 and hi[2] - lo[2] >= 1.0
            and max(hi[:2] - lo[:2]) <= 1.2)


def inside_footprint(frag_xy, body_xy, widen, hull):
    """Share of frag points inside the body footprint widened by widen:
    axis-aligned box, or (hull=True) the convex hull of the body points."""
    if hull and len(body_xy) >= 3:
        try:
            eq = ConvexHull(body_xy).equations      # n.x + c <= 0 inside
        except QhullError:
            eq = None
        if eq is not None:
            dist = frag_xy @ eq[:, :2].T + eq[:, 2]
            return float((dist.max(axis=1) <= widen).mean())
    lo, hi = body_xy.min(axis=0), body_xy.max(axis=0)
    return float(np.all((frag_xy >= lo - widen) & (frag_xy <= hi + widen),
                        axis=1).mean())


def fragment_parent(frag, bodies, ratio, contained, gap, widen=0.3,
                    hull=False):
    """Index into bodies of the cluster frag belongs to, or None."""
    if standing_column(frag):
        return None
    best, best_share = None, 0.0
    for k, body in bodies:
        if len(frag) > ratio * len(body):
            continue
        inside = inside_footprint(frag[:, :2], body[:, :2], widen, hull)
        if inside < contained or inside <= best_share:
            continue
        if cKDTree(body).query(frag, k=1)[0].min() > gap:
            continue
        best, best_share = k, inside
    return best


def merge_fragments(groups, world, ratio, contained, gap, hull=False):
    """Contained-fragment merge of index groups (one pass, no chains)."""
    order = sorted(range(len(groups)), key=lambda i: -len(groups[i]))
    parent = {}
    roots = []
    for i in order:                       # large first: parents are roots
        pts = world[groups[i]]
        p = fragment_parent(pts, [(k, world[groups[k]]) for k in roots],
                            ratio, contained, gap, hull=hull)
        if p is None:
            roots.append(i)
        else:
            parent[i] = p
    merged = {r: [groups[r]] for r in roots}
    for i, p in parent.items():
        merged[p].append(groups[i])
    return [np.concatenate(merged[r]) for r in sorted(roots)], len(parent)


class MergingExtractor(EuclideanClusterExtractor):
    """Euclidean clusters + contained-fragment merge (world geometry)."""
    pose = None
    ratio, contained, gap, hull = 0.3, 0.8, 1.2, False
    merges = 0
    ms = []

    def extract(self, points, context=None):
        start = time.perf_counter()
        groups = super().extract(points, context)
        world = self.pose.to_reference(points)
        groups, n = merge_fragments(groups, world, self.ratio,
                                    self.contained, self.gap, self.hull)
        MergingExtractor.merges += n
        MergingExtractor.ms.append((time.perf_counter() - start) * 1e3)
        return groups


class TimedExtractor(EuclideanClusterExtractor):
    ms = []

    def extract(self, points, context=None):
        start = time.perf_counter()
        groups = super().extract(points, context)
        TimedExtractor.ms.append((time.perf_counter() - start) * 1e3)
        return groups


VARIANTS = {
    "B0": dict(extractor=TimedExtractor, tolerance=0.6),
    "T8": dict(extractor=TimedExtractor, tolerance=0.8),
    "C1": dict(extractor=MergingExtractor, tolerance=0.6, ratio=0.3,
               contained=0.8),
    "C1b": dict(extractor=MergingExtractor, tolerance=0.6, ratio=0.5,
                contained=0.75),
    "C1h": dict(extractor=MergingExtractor, tolerance=0.6, ratio=0.5,
                contained=0.75, hull=True),
    "C2": dict(extractor=TimedExtractor, tolerance=0.6, group=True,
               ratio=0.5, contained=0.75),
}


class Grouper:
    """Object-level link of persistent fragment tracks (variant C2)."""

    def __init__(self, ratio, contained, gap, persist=3):
        self.ratio, self.contained, self.gap = ratio, contained, gap
        self.persist = persist
        self.count = defaultdict(int)       # (frag tid, parent tid) -> n
        self.group = {}                     # tid -> group tid

    def update(self, ids, object_points):
        seen = set()
        confirmed = [(k, t) for k, t in enumerate(ids) if t is not None]
        order = sorted(confirmed, key=lambda kt: -len(object_points[kt[0]]))
        roots = []
        for k, tid in order:
            p = fragment_parent(object_points[k],
                                [(j, object_points[j]) for j, _ in roots],
                                self.ratio, self.contained, self.gap)
            if p is None:
                roots.append((k, tid))
                continue
            parent_tid = ids[p]
            key = (tid, parent_tid)
            self.count[key] += 1
            seen.add(key)
            if self.count[key] >= self.persist:
                self.group[tid] = self.group.get(parent_tid, parent_tid)
        for key in list(self.count):
            if key not in seen:
                del self.count[key]
                if self.group.get(key[0]) is not None and key[0] in [
                        t for _, t in confirmed]:
                    self.group.pop(key[0], None)
        return [None if t is None else self.group.get(t, t) for t in ids]


def run_variant(pose, scene, frames, key, spec):
    config = EvalConfig(cluster=ClusterParams(tolerance=spec["tolerance"]))
    original = eval_objects_gt.EuclideanClusterExtractor
    eval_objects_gt.EuclideanClusterExtractor = spec["extractor"]
    spec["extractor"].ms = []
    if spec["extractor"] is MergingExtractor:
        MergingExtractor.pose = pose
        MergingExtractor.ratio = spec["ratio"]
        MergingExtractor.contained = spec["contained"]
        MergingExtractor.gap = 1.2
        MergingExtractor.hull = spec.get("hull", False)
        MergingExtractor.merges = 0
    try:
        return _run(pose, scene, frames, config, spec)
    finally:
        eval_objects_gt.EuclideanClusterExtractor = original


def _run(pose, scene, frames, config, spec):
    sensor = pose.to_reference(np.zeros((1, 3)))[0]
    tracker = ObjectTracker(TrackerParams())
    grouper = (Grouper(spec["ratio"], spec["contained"], 1.2)
               if spec.get("group") else None)
    records_all, stats_all = [], []
    entries = defaultdict(list)
    split_frames = defaultdict(list)
    merged_events = 0
    moving = {g["object_id"] for g in scene["ground_truth"]
              if g.get("moving")}
    for index, frame in enumerate(frames):
        boxes = gt_boxes(frame["gt"], scene)
        records, stats, objects, matches = evaluate_frame(
            frame["points"], boxes, pose, config, sensor,
            return_objects=True)
        records_all += records
        stats_all.append(stats)
        keep = np.flatnonzero(height_band_mask(
            frame["points"], pose, config.crop_min_z, config.crop_max_z))
        world = pose.to_reference(frame["points"][keep])
        object_points = [world[o.point_indices] for o in objects]
        stamp = frame["gt"]["lidar_stamps_ns"][-1]
        ids = tracker.update(stamp, [o.center_reference for o in objects],
                             "world")
        reported = grouper.update(ids, object_points) if grouper else ids
        centers = {gid: c.mean(axis=0) for gid, _, c in boxes}
        for rec in records:
            gid = rec["id"]
            if rec["status"] == "no_points":
                continue
            k = matches.get(gid)
            entry = {"index": index, "stamp_ns": stamp,
                     "class": rec["class"], "status": rec["status"],
                     "detected": k is not None,
                     "track_id": reported[k] if k is not None else None,
                     "gt_center": centers[gid].tolist()}
            if k is not None and ids[k] is not None:
                entry["track_velocity"] = \
                    tracker.tracks[ids[k]].velocity.tolist()
            entries[gid].append(entry)
        per_gt, merged = frame_split(object_points, boxes, matches)
        merged_events += len(merged)
        cls = {gid: c for gid, c, _ in boxes}
        for gid, rec in per_gt.items():
            body_tid = (reported[rec["body"]] if rec["body"] is not None
                        else None)
            for frag in rec["fragments"]:
                tid = reported[frag["object"]]
                frag["track_id"] = None if tid == body_tid else tid
            rec.update({"index": index, "stamp_ns": stamp,
                        "class": cls[gid], "body_track_id": body_tid})
            split_frames[gid].append(rec)

    obj_summary = summarize(records_all, stats_all)
    tracking = {gid: summarize_object(items) for gid, items in
                entries.items()}
    split = {gid: summarize_gt(items) for gid, items in split_frames.items()}
    vel = [v for gid, o in tracking.items() if gid in moving
           for v in o["velocity_errors"]]
    vel_frames = {}
    for gid in moving & set(entries):
        items = entries[gid]
        for k in range(1, len(items) - 1):
            a, b, e = items[k - 1], items[k + 1], items[k]
            if "track_velocity" not in e:
                continue
            dt = (b["stamp_ns"] - a["stamp_ns"]) * 1e-9
            gv = (np.subtract(b["gt_center"], a["gt_center"]) / dt)[:2]
            if np.linalg.norm(gv) >= 0.3:
                vel_frames[e["index"]] = float(np.linalg.norm(
                    np.subtract(e["track_velocity"][:2], gv)))
    ms = spec["extractor"].ms
    return {
        "observable": obj_summary["observable"],
        "status": obj_summary["status"],
        "ok_rate": obj_summary["ok_rate"],
        "err_xy_bbox": obj_summary.get("err_xy_bbox"),
        "err_distance": obj_summary.get("err_distance"),
        "objects_per_frame": obj_summary.get("objects_per_frame"),
        "id_switches": sum(o["id_switches"] for o in tracking.values()),
        "fragments": sum(o["fragments"] for o in tracking.values()),
        "tracked": sum(o["tracked"] for o in tracking.values()),
        "detected": sum(o["detected"] for o in tracking.values()),
        "split_frames": sum(o["split_frames"] for o in split.values()),
        "fragments_total": sum(o["fragments_total"] for o in split.values()),
        "separate_fragment_tracks": sum(len(o["fragment_track_ids"])
                                        for o in split.values()),
        "fragment_track_frames": sum(
            o["frames_with_confirmed_fragment_track"]
            for o in split.values()),
        "body_fragment_swaps": sum(len(o["body_fragment_swaps"])
                                   for o in split.values()),
        "gt_merged_events": merged_events,
        "fragment_merges": (MergingExtractor.merges
                            if spec["extractor"] is MergingExtractor
                            else None),
        "velocity_error_moving": (
            {"n": len(vel), "p50": float(np.median(vel)),
             "p95": float(np.percentile(vel, 95))} if vel else None),
        "velocity_error_frames": vel_frames,
        "extract_ms": ({"p50": float(np.median(ms)),
                        "p95": float(np.percentile(ms, 95))} if ms else None),
        "per_object": {gid: {
            "id_switches": tracking[gid]["id_switches"],
            "fragments": tracking[gid]["fragments"],
            "split_frames": split[gid]["split_frames"],
            "observable": split[gid]["observable_frames"],
            "fragment_tracks": split[gid]["fragment_track_ids"],
            "swaps": len(split[gid]["body_fragment_swaps"]),
        } for gid in split},
    }


def synthetic(kind, center, normal, rng):
    """Points of a person / cone plane facing direction normal (xy)."""
    width, height, step = ((0.45, 1.7, 0.12) if kind == "person"
                           else (0.3, 0.7, 0.1))
    tangent = np.array([-normal[1], normal[0], 0.0])
    pts = []
    for a in np.arange(-width / 2, width / 2 + 1e-6, step):
        for z in np.arange(0.05, height, step):
            pts.append(center + tangent * a + np.array([0, 0, z])
                       + rng.normal(0, 0.01, 3))
    return np.array(pts)


def inject_test(pose, scene, frames, variants, n_frames=4, seed=0):
    rng = np.random.default_rng(seed)
    sensor = pose.to_reference(np.zeros((1, 3)))[0]
    config = EvalConfig()
    picks = np.linspace(0, len(frames) - 1, min(n_frames, len(frames)),
                        dtype=int)
    out = {key: defaultdict(lambda: defaultdict(int)) for key in variants}
    for idx in picks:
        frame = frames[idx]
        keep = np.flatnonzero(height_band_mask(
            frame["points"], pose, config.crop_min_z, config.crop_max_z))
        cloud = frame["points"][keep]
        for gid, cls, corners in gt_boxes(frame["gt"], scene):
            if not any(v in cls for v in VEHICLE) and cls not in (
                    "minibus", "fire_pumper", "ambulance_staria",
                    "suv_sportage", "suv_sorento", "van_starex",
                    "van_staria", "sedan_mid", "car_sedan"):
                continue
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            c = (lo + hi) / 2
            to_sensor = sensor[:2] - c[:2]
            # Front: the box face toward the sensor (x), and the +y side.
            fx = lo[0] if to_sensor[0] < 0 else hi[0]
            sx = -1.0 if to_sensor[0] < 0 else 1.0
            for gap in (0.4, 0.7, 1.0, 1.5):
                spots = {
                    "front": (np.array([fx + sx * (gap + 0.05), c[1], 0.0]),
                              np.array([sx, 0.0])),
                    "side": (np.array([c[0], hi[1] + gap + 0.05, 0.0]),
                             np.array([0.0, 1.0])),
                }
                for where, (pos, normal) in spots.items():
                    for kind in ("person", "cone"):
                        syn = synthetic(kind, pos, normal, rng)
                        pts = np.vstack([cloud, pose.to_cloud(syn)])
                        mark = np.zeros(len(pts), bool)
                        mark[len(cloud):] = True
                        for key, spec in variants.items():
                            res = classify_injected(pts, mark, pose, spec)
                            out[key][f"{kind}_{where}_{gap}"][res] += 1
    return {k: {c: dict(v) for c, v in d.items()} for k, d in out.items()}


def classify_injected(points, mark, pose, spec):
    groups = EuclideanClusterExtractor(ClusterParams(
        tolerance=spec["tolerance"])).extract(points)
    if spec["extractor"] is MergingExtractor:
        world = pose.to_reference(points)
        groups, _ = merge_fragments(groups, world, spec["ratio"],
                                    spec["contained"], 1.2,
                                    spec.get("hull", False))
    n = mark.sum()
    for g in groups:
        inj = mark[g].sum()
        if inj >= 0.8 * n:
            return "separate" if inj / len(g) >= 0.8 else "absorbed"
    return "absorbed_or_lost"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--variants", default="B0,T8,C1,C1b,C1h,C2")
    parser.add_argument("--inject", action="store_true")
    parser.add_argument("--json")
    args = parser.parse_args()
    chosen = {k: VARIANTS[k] for k in args.variants.split(",")}
    results = {}
    for path in args.paths:
        for bag in bag_dirs(path):
            name = os.path.basename(bag)
            try:
                pose, scene, frames = load_preprocessed(
                    bag, gt_frames="scan")
            except UnsupportedBag as exc:
                print(f"skipped: {exc}", flush=True)
                continue
            results[name] = {}
            for key, spec in chosen.items():
                r = run_variant(pose, scene, frames, key, spec)
                results[name][key] = r
                print(f"{name[:30]} {key}: ok {r['ok_rate']:.3f} "
                      f"idsw {r['id_switches']} frag {r['fragments']} "
                      f"split {r['split_frames']} "
                      f"fragTracks {r['separate_fragment_tracks']} "
                      f"swaps {r['body_fragment_swaps']} "
                      f"gtMerged {r['gt_merged_events']} "
                      f"merges {r['fragment_merges']} "
                      f"ms {r['extract_ms']}", flush=True)
            if args.inject:
                results[name]["inject"] = inject_test(pose, scene, frames,
                                                      chosen)
                print(f"{name[:30]} inject: "
                      f"{json.dumps(results[name]['inject'])}", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
