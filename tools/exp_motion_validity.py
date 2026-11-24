#!/usr/bin/env python3
"""EXPERIMENT: velocity validity variants vs the current rules, offline.

Not production code: lidar/object_motion.py and object_tracker.py are not
changed. Variants are MotionEstimator subclasses / parameter sets defined
here and run on cached detections (tools/eval_motion_gt.detect_bag) with
the same tracker loop as tools/eval_motion_gt.run_bag. LiDAR objects carry
no class, so every rule is class-agnostic (size / kinematics only).

Variants:
    V0 baseline       current MotionParams / TrackerParams
    V1 relax          shape_change_min_m and size_jump_min_m 0.5 -> 0.8 m
    V2 trend          shape_changing tests the footprint TREND over window_s
                      (least-squares slope * window, per edge) instead of
                      the range: periodic changes (gait) pass, monotonic
                      growth (object emerging from occlusion) is held;
                      size_jump_min_m 0.8 m
    V3 transient      baseline + "transient": unknown when the least-squares
                      velocity of the last --short-s (>= 4 samples) differs
                      from the Kalman velocity by more than --short-mismatch
                      (start / stop: the Kalman velocity lags)
    V4 trend+trans    V2 + V3
    V5 relax+trans    V1 + V3
    V6 accel4         baseline validity, tracker accel_std 2 -> 4 m/s^2
                      (also changes association)
    V7 accel4+std     V6 with max_speed_std 0.6 -> 1.0 m/s (the larger
                      process noise raises the Kalman velocity sigma)

Metrics per bag and GT object (path velocity of gt.json, eval_velocity_gt):
    moving_published_share   GT moving frames (no transition) with a
                             published velocity / those frames
    published_err            vector error of published velocities (moving)
    wrong_published          published with error > 0.3 m/s (all phases)
    wrong_moving             state "moving" with error > 0.3 m/s: a wrong
                             velocity that reaches the TTC computation
    t_first_good_after_start first published frame with error <= 0.3 m/s
    wrong_after_start        published with error > 0.3 in 2 s after start
    t_moving_release         last "moving" frame after the GT stop (s)
    false_moving_static      "moving" while the GT object is stationary
                             (before start / after stop, static GT objects)
    background_moving        "moving" frames of confirmed tracks matched to
                             no GT object

Usage:
    python3 tools/exp_motion_validity.py <bag-dir>... --cache-dir DIR \
        [--variants V0,V3] [--json out.json]
"""

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lidar"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_motion_gt  # noqa: E402
from eval_motion_gt import bag_list, detect_bag  # noqa: E402
from eval_objects_gt import UnsupportedBag  # noqa: E402
from eval_velocity_gt import PathMotion, frame_rows  # noqa: E402
from object_motion import (  # noqa: E402
    MotionEstimator, MotionParams, ObjectMotion, TtcParams)
from object_tracker import TrackerParams  # noqa: E402

WRONG = 0.3


def _ls_slope(t, values):
    A = np.stack([t, np.ones_like(t)], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, values, rcond=None)
    return coeffs[0]


class TrendMotion(MotionEstimator):
    """shape_changing on the footprint trend, not its range."""

    def _shape_changing(self, history):
        p = self.params
        samples = list(history.samples)
        if len(samples) < 4:
            return False
        t = np.array([(s - samples[-1][0]) * 1e-9 for s, _, _ in samples])
        if t.max() - t.min() < p.window_s / 2:
            return False
        sizes = np.array([size for _, _, size in samples])
        trend = np.abs(_ls_slope(t, sizes)) * p.window_s
        limit = np.maximum(p.shape_change_ratio * np.median(sizes, axis=0),
                           p.shape_change_min_m)
        return bool(np.any(trend > limit))


def transient_mixin(base, short_s, mismatch, min_samples=4):
    class Transient(base):
        def _judge(self, track, stamp_ns, world, measured):
            result = super()._judge(track, stamp_ns, world, measured)
            if result.state == "unknown" or result.velocity is None:
                return result
            history = self.histories.get(track.id)
            if history is None:
                return result
            t0 = stamp_ns - int(short_s * 1e9)
            pts = [(s, xy) for s, xy, _ in history.samples if s >= t0]
            if len(pts) < min_samples:
                return result
            t = np.array([(s - stamp_ns) * 1e-9 for s, _ in pts])
            if t.max() - t.min() < short_s * 0.6:
                return result
            v_short = _ls_slope(t, np.array([xy for _, xy in pts]))
            if np.linalg.norm(v_short - result.velocity) <= mismatch:
                return result
            return self._unknown(track, "transient", stamp_ns, measured)
    return Transient


def variants(args):
    base_m, base_t = MotionParams(), TrackerParams()
    relax = replace(base_m, shape_change_min_m=0.8, size_jump_min_m=0.8)
    trend = replace(base_m, size_jump_min_m=0.8)
    trans = lambda cls: transient_mixin(  # noqa: E731
        cls, args.short_s, args.short_mismatch, args.short_min)
    return {
        "V0": ("baseline", MotionEstimator, base_m, base_t),
        "V1": ("relax", MotionEstimator, relax, base_t),
        "V2": ("trend", TrendMotion, trend, base_t),
        "V3": ("transient", trans(MotionEstimator), base_m, base_t),
        "V4": ("trend+transient", trans(TrendMotion), trend, base_t),
        "V5": ("relax+transient", trans(MotionEstimator), relax, base_t),
        "V6": ("accel4", MotionEstimator, base_m,
               replace(base_t, accel_std=4.0)),
        "V7": ("accel4+std", MotionEstimator,
               replace(base_m, max_speed_std=1.0),
               replace(base_t, accel_std=4.0)),
    }


def run_variant(data, cls, motion_params, tracker_params):
    original = eval_motion_gt.MotionEstimator
    eval_motion_gt.MotionEstimator = cls
    try:
        return eval_motion_gt.run_bag(data, tracker_params, motion_params,
                                      TtcParams())
    finally:
        eval_motion_gt.MotionEstimator = original


def stats(values):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return {"n": 0}
    return {"n": int(len(v)), "mae": round(float(v.mean()), 4),
            "p50": round(float(np.median(v)), 4),
            "p95": round(float(np.percentile(v, 95)), 4),
            "max": round(float(v.max()), 4)}


def moving_metrics(rows, path):
    tracked = [r for r in rows if "kalman" in r]
    moving = [r for r in tracked if r["phase"] == "moving"
              and r["segment"] != "transition"]
    pub_moving = [r for r in moving if r["published"] is not None]
    published = [r for r in tracked if r["published"] is not None]
    start = path.epoch_ns
    stop = path.epoch_ns + path.duration_ns
    good_after = [r for r in published if r["stamp_ns"] > start
                  and r["err_vector"] <= WRONG and r["phase"] == "moving"]
    moving_after_stop = [r for r in tracked if r["stamp_ns"] >= stop
                         and r["state"] == "moving"]
    stationary = [r for r in tracked if r["phase"] != "moving"]
    return {
        "moving_frames": len(moving),
        "moving_published_share": (round(len(pub_moving) / len(moving), 3)
                                   if moving else None),
        "published_err": stats([r["err_vector"] for r in pub_moving]),
        "wrong_published": sum(r["err_vector"] > WRONG for r in published),
        "wrong_moving": sum(r["err_vector"] > WRONG and r["state"] == "moving"
                            for r in tracked),
        "t_first_good_after_start_s": (
            round((min(r["stamp_ns"] for r in good_after) - start) * 1e-9, 2)
            if good_after else None),
        "wrong_after_start": sum(
            r["err_vector"] > WRONG for r in published
            if 0 < (r["stamp_ns"] - start) * 1e-9 <= 2.0),
        "t_moving_release_s": (
            round((max(r["stamp_ns"] for r in moving_after_stop) - stop)
                  * 1e-9, 2) if moving_after_stop else 0.0),
        "false_moving_static": sum(r["state"] == "moving"
                                   for r in stationary),
        "stationary_frames": len(stationary),
        "unknown_reasons": dict(Counter(r["reason"] for r in tracked
                                        if r["state"] == "unknown")),
        "rows": [{k: r.get(k) for k in (
            "index", "stamp_ns", "phase", "segment", "v_path", "kalman",
            "published", "state", "reason", "err_vector")}
            for r in tracked],
    }


def static_metrics(entries):
    states = Counter(e["state"] for e in entries if "state" in e)
    speeds = [float(np.linalg.norm(e["raw_velocity"])) for e in entries
              if e.get("state") not in (None, "unknown")]
    return {"tracked_frames": sum(states.values()),
            "false_moving": states.get("moving", 0),
            "published_speed": stats(speeds), "states": dict(states)}


def evaluate_bag(bag, args, chosen):
    data = detect_bag(bag, args.cache_dir, gt_frames="scan")
    scene = data["scene"]
    paths = {g["object_id"]: PathMotion(g, scene["eval_frame"])
             for g in scene["ground_truth"] if g.get("motion")}
    row_args = SimpleNamespace(init_s=1.0, start_s=1.0, stop_s=1.5,
                               gap_ms=150.0, gap_hold_s=0.5)
    out = {}
    for key, (name, cls, mparams, tparams) in chosen.items():
        result = run_variant(data, cls, mparams, tparams)
        objects = {}
        for gid, entries in result["entries"].items():
            if gid in paths:
                objects[gid] = moving_metrics(
                    frame_rows(entries, paths[gid], row_args), paths[gid])
            else:
                objects[gid] = static_metrics(entries)
        ids = Counter()
        for gid, entries in result["entries"].items():
            last, switches = None, 0
            for e in entries:
                tid = e.get("track_id")
                if tid is not None:
                    switches += last is not None and tid != last
                    last = tid
            ids[gid] = switches
        out[key] = {
            "name": name,
            "background_moving": result["background_states"].get("moving",
                                                                  0),
            "background_states": dict(result["background_states"]),
            "resets": result["resets"],
            "gt_id_switches": dict(ids),
            "gt_tracked_frames": {gid: sum(e.get("track_id") is not None
                                           for e in entries)
                                  for gid, entries in
                                  result["entries"].items()},
            "objects": objects,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--variants", default="V0,V1,V2,V3,V4,V5,V6,V7")
    parser.add_argument("--short-s", type=float, default=0.5)
    parser.add_argument("--short-mismatch", type=float, default=0.4)
    parser.add_argument("--short-min", type=int, default=4,
                        help="minimum samples in the short window")
    parser.add_argument("--json")
    args = parser.parse_args()

    allv = variants(args)
    chosen = {k: allv[k] for k in args.variants.split(",")}
    results = {}
    for bag in bag_list(args.paths):
        name = os.path.basename(os.path.normpath(bag))
        try:
            results[name] = evaluate_bag(bag, args, chosen)
        except UnsupportedBag as exc:
            print(f"skipped: {exc}", flush=True)
            continue
        for key, r in results[name].items():
            parts = []
            for gid, o in r["objects"].items():
                if "moving_published_share" in o:
                    parts.append(
                        f"{gid}: pub {o['moving_published_share']} "
                        f"err p95 {o['published_err'].get('p95')} "
                        f"wrongMov {o['wrong_moving']} "
                        f"tGood {o['t_first_good_after_start_s']} "
                        f"release {o['t_moving_release_s']} "
                        f"falseMov {o['false_moving_static']}")
                elif o["false_moving"]:
                    parts.append(f"{gid}: static falseMov "
                                 f"{o['false_moving']}")
            print(f"{name[:30]} {key} {r['name']}: bgMoving "
                  f"{r['background_moving']} | " + " | ".join(parts),
                  flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
