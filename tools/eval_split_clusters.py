#!/usr/bin/env python3
"""Split / merge metrics per GT object, offline (separate result file).

Same preprocessing, clustering, GT match and tracker as
tools/eval_objects_gt.py and tools/eval_tracking_gt.py (one GT frame per
scan); the existing evaluation outputs are not changed.

Per frame and GT box (frame_split):
    candidate  a cluster with >= min_points points inside the box (with the
               evaluation margin) AND >= purity of its own points inside
               the box: mostly made of this object's points, not a
               structure passing through the box
    body       the cluster matched to the GT (eval_objects_gt rule)
    fragment   a candidate that is not the body
    merged     a cluster holding >= merge_share of the in-box points of two
               or more GT boxes (and >= min_points in each)
Per GT object over the bag:
    clusters_per_frame   histogram of candidates (+ body) per observable
                         frame: 0, 1, 2, 3+
    split_frame_ratio    frames with >= 1 fragment / observable frames
    fragment_point_share mean share of the clustered in-box points that
                         are in fragments (frames with fragments)
    fragment_cells       fragments grouped by position in the box frame
                         (along the box's first edge, second edge, height;
                         cell --cell m): frame share, median size, bottom
                         height and points. persistent = frame share >=
                         --persist; the rest is "transient"
    fragment_tracks      distinct confirmed track IDs seen on fragments,
                         frames where a fragment had a confirmed track
    body_fragment_swaps  changes of the body's track ID to an ID that a
                         fragment of this GT carried before: the GT match
                         moved to another piece, the cause of "ok"-status
                         ID switches (one per change, not per frame)
    relation             per fragment, to the body cluster: minimum 3D
                         point gap, horizontal gap of the two footprints
                         (axis-aligned boxes), vertical gap (> 0 = one above
                         the other), share of fragment points whose xy lies
                         inside the body's xy box widened by 0.3 m
                         ("contained"), points relative to the body
Per bag: merged events (frame index, stamp, GT IDs, shares).

Usage:
    python3 tools/eval_split_clusters.py <scenario-or-bag-dir>... \
        --json split.json [--min-points 5] [--purity 0.5] ...
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lidar"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_objects_gt import (  # noqa: E402
    BOX_MARGIN,
    MIN_OBSERVED,
    EvalConfig,
    UnsupportedBag,
    bag_dirs,
    evaluate_frame,
    gt_boxes,
    inside_box,
    load_preprocessed,
)
from object_estimator import height_band_mask  # noqa: E402
from object_tracker import ObjectTracker, TrackerParams  # noqa: E402


def box_frame(corners):
    """(origin, unit axes 3x3, edge lengths) of a GT box."""
    origin = corners[0]
    axes, lengths = [], []
    for corner in (corners[1], corners[3], corners[4]):
        axis = corner - origin
        lengths.append(float(np.linalg.norm(axis)))
        axes.append(axis / lengths[-1])
    return origin, np.array(axes), np.array(lengths)


def relation(fragment, body, widen=0.3):
    """Spatial relation of a fragment's points to the body's points."""
    gap = float(cKDTree(body).query(fragment, k=1)[0].min())
    fmin, fmax = fragment.min(axis=0), fragment.max(axis=0)
    bmin, bmax = body.min(axis=0), body.max(axis=0)
    h = np.maximum(0.0, np.maximum(bmin[:2] - fmax[:2], fmin[:2] - bmax[:2]))
    v = max(fmin[2] - bmax[2], bmin[2] - fmax[2])
    inside = np.all((fragment[:, :2] >= bmin[:2] - widen)
                    & (fragment[:, :2] <= bmax[:2] + widen), axis=1)
    return {"gap_3d": gap, "gap_h": float(np.hypot(*h)), "gap_v": float(v),
            "contained": float(inside.mean()),
            "points_ratio": float(len(fragment) / max(len(body), 1))}


def frame_split(object_points, boxes, matches, min_points=5, purity=0.5,
                merge_share=0.3, margin=BOX_MARGIN):
    """Split / merge records of one frame.

    Args:
        object_points: list of (N_i, 3) world points per object (index =
            object id).
        boxes: [(gt_id, class, (8, 3) corners)].
        matches: {gt_id: body object index} (eval_objects_gt match).
    Returns:
        (per_gt, merged): per_gt[gt_id] = {"observable", "body",
        "fragments": [...], "candidates"}, merged = [{"object", "gt_ids",
        "shares"}].
    """
    per_gt, inbox = {}, {}
    for gid, _, corners in boxes:
        counts = np.array([int(inside_box(p, corners, margin).sum())
                           for p in object_points], dtype=int)
        inbox[gid] = counts
        clustered = int(counts.sum())
        sizes = np.array([len(p) for p in object_points])
        candidates = [
            i for i, c in enumerate(counts)
            if c >= min_points and c / max(sizes[i], 1) >= purity
        ]
        body = matches.get(gid)
        origin, axes, lengths = box_frame(np.asarray(corners))
        fragments = []
        for i in candidates:
            if i == body:
                continue
            pts = object_points[i]
            local = (pts - origin) @ axes.T
            fragments.append({
                "object": i,
                "points": int(len(pts)),
                "in_box": int(counts[i]),
                "purity": float(counts[i] / len(pts)),
                "share_of_box": float(counts[i] / max(clustered, 1)),
                "local_center": np.median(local, axis=0).tolist(),
                "size": (pts.max(axis=0) - pts.min(axis=0)).tolist(),
                "bottom_z": float(pts[:, 2].min()),
                **(relation(pts, object_points[body])
                   if body is not None and len(object_points[body])
                   else {}),
            })
        per_gt[gid] = {
            "observable": clustered >= MIN_OBSERVED,
            "body": body,
            "candidates": sorted(set(candidates) | (
                {body} if body is not None else set())),
            "fragments": fragments,
            "clustered_in_box": clustered,
            "box_lengths": lengths.tolist(),
        }

    merged = []
    for i, pts in enumerate(object_points):
        hits, shares = [], []
        for gid, counts in inbox.items():
            total = counts.sum()
            if counts[i] >= min_points and total and (
                    counts[i] / total >= merge_share):
                hits.append(gid)
                shares.append(float(counts[i] / total))
        if len(hits) >= 2:
            merged.append({"object": i, "gt_ids": hits, "shares": shares,
                           "points": int(len(pts))})
    return per_gt, merged


def summarize_gt(frames, cell=0.5, persist=0.2):
    """Bag metrics of one GT object from its per-frame records."""
    observable = [f for f in frames if f["observable"]]
    n = len(observable)
    hist = Counter(min(len(f["candidates"]), 3) for f in observable)
    split = [f for f in observable if f["fragments"]]
    cells = defaultdict(list)
    for f in observable:
        for frag in f["fragments"]:
            key = tuple(int(np.floor(v / cell)) for v in frag["local_center"])
            cells[key].append((f["index"], frag))
    cell_rows = []
    for key, items in cells.items():
        share = len({k for k, _ in items}) / n if n else 0.0
        frags = [fr for _, fr in items]
        cell_rows.append({
            "local_center_med": np.round(np.median(
                [fr["local_center"] for fr in frags], axis=0), 2).tolist(),
            "frame_share": round(share, 3),
            "persistent": share >= persist,
            "size_med": np.round(np.median(
                [fr["size"] for fr in frags], axis=0), 2).tolist(),
            "bottom_z_med": round(float(np.median(
                [fr["bottom_z"] for fr in frags])), 2),
            "points_med": float(np.median([fr["points"] for fr in frags])),
            "purity_med": round(float(np.median(
                [fr["purity"] for fr in frags])), 2),
            "confirmed_track_frames": sum(
                fr.get("track_id") is not None for fr in frags),
            **{key + "_med": round(float(np.median(
                [fr[key] for fr in frags if key in fr])), 3)
               for key in ("gap_3d", "gap_h", "gap_v", "contained",
                           "points_ratio")
               if any(key in fr for fr in frags)},
        })
    cell_rows.sort(key=lambda r: -r["frame_share"])

    frag_tids = [fr.get("track_id") for f in observable
                 for fr in f["fragments"] if fr.get("track_id") is not None]
    seen_fragment_ids, swaps, last_body = set(), [], None
    for f in observable:
        body_tid = f.get("body_track_id")
        if (body_tid is not None and body_tid != last_body
                and body_tid in seen_fragment_ids):
            swaps.append({"index": f["index"], "stamp_ns": f["stamp_ns"],
                          "track_id": body_tid})
        if body_tid is not None:
            last_body = body_tid
        seen_fragment_ids |= {fr["track_id"] for fr in f["fragments"]
                              if fr.get("track_id") is not None}
    return {
        "class": frames[0]["class"] if frames else None,
        "observable_frames": n,
        "body_frames": sum(f["body"] is not None for f in observable),
        "clusters_per_frame": {str(k) if k < 3 else "3+": hist.get(k, 0)
                               for k in range(4)},
        "split_frames": len(split),
        "split_frame_ratio": round(len(split) / n, 3) if n else None,
        "fragments_total": sum(len(f["fragments"]) for f in observable),
        "fragment_point_share": round(float(np.mean([
            sum(fr["in_box"] for fr in f["fragments"])
            / max(f["clustered_in_box"], 1) for f in split])), 3)
        if split else 0.0,
        "persistent_fragment_cells": sum(r["persistent"] for r in cell_rows),
        "transient_fragment_cells": sum(not r["persistent"]
                                        for r in cell_rows),
        "fragment_cells": cell_rows,
        "fragment_track_ids": sorted(set(frag_tids)),
        "frames_with_confirmed_fragment_track": len({
            f["index"] for f in observable
            if any(fr.get("track_id") is not None for fr in f["fragments"])}),
        "body_fragment_swaps": swaps,
        "split_frame_indices": [f["index"] for f in split],
    }


def evaluate_bag(bag, args):
    pose, scene, frames = load_preprocessed(bag, gt_frames="scan")
    return evaluate_loaded(pose, scene, frames, args)


def evaluate_loaded(pose, scene, frames, args, config=None):
    """evaluate_bag on preprocessed frames (config: EvalConfig)."""
    sensor = pose.to_reference(np.zeros((1, 3)))[0]
    config = config or EvalConfig()
    tracker = ObjectTracker(TrackerParams())
    per_gt_frames = defaultdict(list)
    merged_events = []
    for index, frame in enumerate(frames):
        boxes = gt_boxes(frame["gt"], scene)
        _, _, objects, matches = evaluate_frame(
            frame["points"], boxes, pose, config, sensor,
            return_objects=True)
        # Same crop as evaluate_frame: object point indices refer to it.
        keep = np.flatnonzero(height_band_mask(
            frame["points"], pose, config.crop_min_z, config.crop_max_z))
        world = pose.to_reference(frame["points"][keep])
        object_points = [world[o.point_indices] for o in objects]
        stamp = frame["gt"]["lidar_stamps_ns"][-1]
        ids = tracker.update(stamp, [o.center_reference for o in objects],
                             "world")
        per_gt, merged = frame_split(
            object_points, boxes, matches, args.min_points, args.purity,
            args.merge_share)
        classes = {gid: cls for gid, cls, _ in boxes}
        for gid, rec in per_gt.items():
            for frag in rec["fragments"]:
                frag["track_id"] = ids[frag["object"]]
            rec.update({
                "index": index, "stamp_ns": stamp, "class": classes[gid],
                "body_track_id": (ids[rec["body"]]
                                  if rec["body"] is not None else None),
            })
            per_gt_frames[gid].append(rec)
        for event in merged:
            event.update({"index": index, "stamp_ns": stamp})
            merged_events.append(event)

    objects = {gid: summarize_gt(items, args.cell, args.persist)
               for gid, items in per_gt_frames.items()}
    return {
        "frames": len(frames),
        "gt_objects": len(objects),
        "split_frames_total": sum(o["split_frames"] for o in objects.values()),
        "observable_total": sum(o["observable_frames"]
                                for o in objects.values()),
        "persistent_fragment_cells": sum(o["persistent_fragment_cells"]
                                         for o in objects.values()),
        "confirmed_fragment_tracks": sum(len(o["fragment_track_ids"])
                                         for o in objects.values()),
        "body_fragment_swaps": sum(len(o["body_fragment_swaps"])
                                   for o in objects.values()),
        "merged_frames": len({e["index"] for e in merged_events}),
        "merged_events": merged_events,
        "objects": objects,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json")
    parser.add_argument("--min-points", type=int, default=5)
    parser.add_argument("--purity", type=float, default=0.5)
    parser.add_argument("--merge-share", type=float, default=0.3)
    parser.add_argument("--cell", type=float, default=0.5)
    parser.add_argument("--persist", type=float, default=0.2)
    args = parser.parse_args()

    results = {}
    for path in args.paths:
        for bag in bag_dirs(path):
            name = os.path.basename(bag)
            try:
                result = evaluate_bag(bag, args)
            except UnsupportedBag as exc:
                print(f"skipped: {exc}", flush=True)
                continue
            results[name] = result
            print(f"{name}: frames={result['frames']} "
                  f"split={result['split_frames_total']}/"
                  f"{result['observable_total']} "
                  f"persistent_cells={result['persistent_fragment_cells']} "
                  f"fragment_tracks={result['confirmed_fragment_tracks']} "
                  f"swaps={result['body_fragment_swaps']} "
                  f"merged_frames={result['merged_frames']}", flush=True)
            for gid, o in result["objects"].items():
                if o["split_frames"] or o["body_fragment_swaps"]:
                    print(f"  {gid} ({o['class']}): split "
                          f"{o['split_frames']}/{o['observable_frames']} "
                          f"clusters/frame {o['clusters_per_frame']} "
                          f"cells persistent/transient "
                          f"{o['persistent_fragment_cells']}/"
                          f"{o['transient_fragment_cells']} "
                          f"fragment_tracks {o['fragment_track_ids']} "
                          f"swaps {len(o['body_fragment_swaps'])}")
    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
