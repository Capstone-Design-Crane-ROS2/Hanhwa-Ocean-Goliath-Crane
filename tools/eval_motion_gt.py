#!/usr/bin/env python3
"""Evaluate velocity validity and crane-face TTC against the GT, offline.

Detection is the same as tools/eval_objects_gt.py (point-based GT match);
tracking uses ObjectTracker with the frame's real stamp; motion and TTC use
object_motion.py. The boundary is the dataset's crane face plane from the
bag's gt.json eval_frame (x_eval = dx * (world_x - ox)).

Per GT object frame (only frames where the GT object is matched and tracked):
    raw velocity error    |v_kalman - v_gt| (xy), GT velocity by central
                          difference of GT centers
    valid velocity error  same, only where the motion state is not unknown
    two-frame difference  |(c_k - c_k-1) / dt - v_gt| of the measured
                          centers, for comparison only
    state                 static GT (|v_gt| < 0.3 m/s, or no motion in
                          gt.json) reported "moving" = false motion
Over all published objects (GT or background):
    motion states, TTC statuses; a "valid" TTC is a false approach alarm
    when no GT object approaches the face (true for every bag in the
    dataset: the only moving object, the S7 truck, moves parallel to it).

Finite TTC (--virtual-plane PX PY NX NY): the dataset has no object that
approaches the crane face, so finite TTC values cannot be checked against
the real face. A hypothetical plane placed across a moving object's path
(e.g. S7: 0 30 0 -1, the truck drives +y at 2 m/s toward y = 30) gives a
finite GT TTC = GT box distance / GT closing speed, computed from the GT
boxes exactly like the estimate. This checks the TTC computation on real
detections; it is not a crane result.

Detections can be cached (--cache-dir) since clustering dominates the run
time; tracking/motion parameters can then be swept quickly.

Usage:
    python3 tools/eval_motion_gt.py <scenario-or-bag-dir>... \
        [--cache-dir DIR] [--json out.json] [--drop-rate 0.2] \
        [--no-validity]
"""

import argparse
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lidar"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crane_boundary import PlaneBoundary  # noqa: E402
from object_motion import (  # noqa: E402
    MotionEstimator,
    MotionParams,
    TtcParams,
    assess_boundary,
)
from object_tracker import ObjectTracker, TrackerParams  # noqa: E402


MOVING_SPEED = 0.3
DETECTED = ("ok", "split")


def detect_bag(bag_dir, cache_dir=None):
    """Per-frame detections of a bag (cached when cache_dir is given)."""
    name = os.path.basename(os.path.normpath(bag_dir))
    path = os.path.join(cache_dir, name + ".pkl") if cache_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    from eval_objects_gt import (
        EvalConfig, evaluate_frame, gt_boxes, load_preprocessed,
    )
    pose, scene, frames = load_preprocessed(bag_dir)
    sensor = pose.to_reference(np.zeros((1, 3)))[0]
    out = []
    for frame in frames:
        boxes = gt_boxes(frame["gt"], scene)
        records, _, objects, matches = evaluate_frame(
            frame["points"], boxes, pose, EvalConfig(), sensor,
            return_objects=True,
        )
        out.append({
            "stamp_ns": frame["gt"]["lidar_stamps_ns"][-1],
            "boxes": boxes,
            "records": records,
            "matches": matches,
            "objects": [dict(
                center_reference=o.center_reference, bbox_min=o.bbox_min,
                bbox_max=o.bbox_max, size=o.size,
                point_count=o.point_count, distance=o.distance,
            ) for o in objects],
        })
    data = {"pose": (pose.rotation_xyzw, pose.translation),
            "scene": scene, "frames": out}
    if path:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(data, handle)
    return data


def face_boundary(scene):
    """(PlaneBoundary, dx, ox) from gt.json eval_frame."""
    frame = scene["eval_frame"]
    dx = float(frame["dx"])
    return (PlaneBoundary((frame["ox"], frame["oy"]), (dx, 0.0)), dx,
            float(frame["ox"]))


def moving_gt_ids(scene):
    return {g["object_id"] for g in scene["ground_truth"] if g.get("moving")}


def run_bag(data, tracker_params, motion_params, ttc_params, drop_rate=0.0,
            seed=0, validity=True, boundary=None):
    """Runs tracker + motion + TTC over cached detections.

    validity=False reports every confirmed track as valid (raw Kalman, for
    the before/after comparison): only the stationary/moving split is kept.
    """
    tracker = ObjectTracker(tracker_params)
    motion = MotionEstimator(motion_params if validity else replace(
        motion_params, min_track_age_s=0.0, min_hits=0, max_speed_std=1e9,
        max_speed=1e9, size_jump_ratio=1e9, size_jump_min_m=1e9,
        point_jump_ratio=1e9, point_jump_min=10**9,
        max_window_mismatch=1e9, shape_change_ratio=1e9,
        shape_change_min_m=1e9,
        # A window too short for any sample pair disables the window
        # velocity checks.
        window_s=1e-6,
    ))
    face, dx, ox = face_boundary(data["scene"])
    boundary = boundary or face
    moving_ids = moving_gt_ids(data["scene"])
    rng = np.random.default_rng(seed)
    entries = defaultdict(list)
    states = Counter()
    reasons = Counter()
    ttc_status = Counter()
    background_states = Counter()
    alarms = []
    frames = 0
    for index, frame in enumerate(data["frames"]):
        if drop_rate > 0 and rng.random() < drop_rate:
            continue
        frames += 1
        stamp = frame["stamp_ns"]
        objects = [SimpleNamespace(**o) for o in frame["objects"]]
        ids = tracker.update(stamp, [o.center_reference for o in objects],
                             "world")
        motions = motion.update(tracker, stamp, objects)
        risks = [
            assess_boundary(m, o.bbox_min, o.bbox_max, o.center_reference,
                            boundary, ttc_params)
            for o, m in zip(objects, motions)
        ]
        matched = {v: k for k, v in frame["matches"].items()}
        for k, (m, r) in enumerate(zip(motions, risks)):
            if ids[k] is None:
                continue
            states[m.state] += 1
            if m.reason:
                reasons[m.reason] += 1
            ttc_status[r.status] += 1
            if k not in matched:
                background_states[m.state] += 1
            if r.status == "valid":
                alarms.append({"index": index, "track_id": ids[k],
                               "gt": matched.get(k), "ttc": r.ttc,
                               "distance": r.distance,
                               "closing": r.closing_speed})

        records = {r["id"]: r for r in frame["records"]}
        for gid, cls, corners in frame["boxes"]:
            record = records[gid]
            if record["status"] == "no_points":
                continue
            corners = np.asarray(corners)
            k = frame["matches"].get(gid)
            normal = boundary.normal()
            entry = {
                "index": index, "stamp_ns": stamp, "class": cls,
                "gt_center": corners.mean(axis=0)[:2],
                "gt_face_distance": float(
                    ((corners[:, :2] - boundary.point) @ normal).min()),
                "gt_moving": gid in moving_ids,
                "track_id": ids[k] if k is not None else None,
            }
            if k is not None and ids[k] is not None:
                m, r = motions[k], risks[k]
                entry.update({
                    "center": np.asarray(objects[k].center_reference)[:2],
                    "raw_velocity": m.velocity,
                    "state": m.state, "reason": m.reason,
                    "face_distance": r.distance,
                    "closing": r.closing_speed, "ttc_status": r.status,
                    "ttc": r.ttc, "ttc_range": r.ttc_range,
                })
            entries[gid].append(entry)
    return {
        "frames": frames, "entries": entries, "states": states,
        "reasons": reasons, "ttc_status": ttc_status,
        "background_states": background_states, "alarms": alarms,
        "resets": tracker.resets, "dx": dx,
        "normal": boundary.normal(), "margin": ttc_params.safety_margin_m,
    }


def _central_difference(entries, key):
    out = [None] * len(entries)
    for k in range(1, len(entries) - 1):
        a, b = entries[k - 1], entries[k + 1]
        dt = (b["stamp_ns"] - a["stamp_ns"]) * 1e-9
        if dt > 0:
            out[k] = (np.asarray(b[key]) - np.asarray(a[key])) / dt
    return out


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"n": 0}
    return {"n": int(len(values)), "mae": float(values.mean()),
            "p50": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max())}


def summarize(result):
    """Velocity and state metrics of one bag run (lists, for pooling)."""
    out = defaultdict(list)
    normal = result["normal"]
    for gid, items in result["entries"].items():
        gt_v = _central_difference(items, "gt_center")
        for k, (entry, v_gt) in enumerate(zip(items, gt_v)):
            if "state" not in entry:
                continue
            moving = entry["gt_moving"] and (
                v_gt is not None and np.linalg.norm(v_gt) >= MOVING_SPEED)
            if entry["gt_moving"] and v_gt is None:
                continue  # first / last frame of a moving object
            v_raw = entry["raw_velocity"]
            if moving:
                err = float(np.linalg.norm(v_raw - v_gt))
                out["moving_raw_err"].append(err)
                out["moving_tracked"].append(1)
                if entry["state"] != "unknown":
                    out["moving_valid_err"].append(err)
                out["moving_state"].append(entry["state"])
                prev = items[k - 1]
                dt = (entry["stamp_ns"] - prev["stamp_ns"]) * 1e-9
                if "center" in prev and dt > 0:
                    fd = (entry["center"] - prev["center"]) / dt
                    out["moving_fd_err"].append(
                        float(np.linalg.norm(fd - v_gt)))
                gt_closing = float(-(normal @ v_gt))
                if entry["closing"] is not None:
                    out["closing_err"].append(
                        abs(entry["closing"] - gt_closing))
                remaining = entry["gt_face_distance"] - result["margin"]
                if gt_closing > 0 and remaining > 0:
                    gt_ttc = remaining / gt_closing
                    out["gt_ttc_frames"].append(gt_ttc)
                    if entry["ttc"] is not None:
                        out["ttc_err"].append(entry["ttc"] - gt_ttc)
                        out["ttc_rel_err"].append(
                            abs(entry["ttc"] - gt_ttc) / gt_ttc)
                        low, high = entry["ttc_range"]
                        out["ttc_in_range"].append(
                            low <= gt_ttc <= (high or np.inf))
                out["moving_face_dist_err"].append(
                    entry["face_distance"] - entry["gt_face_distance"])
                out["moving_ttc_status"].append(entry["ttc_status"])
            else:
                speed = float(np.linalg.norm(v_raw))
                out["static_raw_speed"].append(speed)
                out["static_state"].append(entry["state"])
                if entry["state"] != "unknown":
                    out["static_valid_speed"].append(speed)
                out["static_ttc_status"].append(entry["ttc_status"])
    return out


def report(pooled, runs):
    moving_states = Counter(pooled["moving_state"])
    static_states = Counter(pooled["static_state"])
    n_moving = sum(moving_states.values())
    n_static = sum(static_states.values())
    total = lambda key: sum((r[key] for r in runs), Counter())  # noqa: E731
    background = total("background_states")
    alarms = [a for r in runs for a in r["alarms"]]
    return {
        "frames": sum(r["frames"] for r in runs),
        "resets": sum(r["resets"] for r in runs),
        "moving": {
            "tracked_frames": n_moving,
            "raw_kalman_error": _stats(pooled["moving_raw_err"]),
            "valid_velocity_error": _stats(pooled["moving_valid_err"]),
            "two_frame_difference_error": _stats(pooled["moving_fd_err"]),
            "valid_share": (
                1 - moving_states["unknown"] / n_moving if n_moving else None
            ),
            "states": dict(moving_states),
            "closing_speed_error": _stats(pooled["closing_err"]),
            "face_distance_error": _stats(
                np.abs(pooled["moving_face_dist_err"])),
            "face_distance_bias": (
                float(np.mean(pooled["moving_face_dist_err"]))
                if pooled["moving_face_dist_err"] else None),
            "ttc_status": dict(Counter(pooled["moving_ttc_status"])),
            "gt_finite_ttc_frames": len(pooled["gt_ttc_frames"]),
            "ttc_error_s": _stats(np.abs(pooled["ttc_err"])),
            "ttc_bias_s": (float(np.mean(pooled["ttc_err"]))
                           if pooled["ttc_err"] else None),
            "ttc_relative_error": _stats(pooled["ttc_rel_err"]),
            "gt_ttc_inside_range": (
                float(np.mean(pooled["ttc_in_range"]))
                if pooled["ttc_in_range"] else None),
        },
        "static": {
            "tracked_frames": n_static,
            "raw_speed": _stats(pooled["static_raw_speed"]),
            "reported_speed": _stats(pooled["static_valid_speed"]),
            "false_moving": static_states["moving"],
            "false_moving_rate": (
                static_states["moving"] / n_static if n_static else None),
            "states": dict(static_states),
            "ttc_status": dict(Counter(pooled["static_ttc_status"])),
        },
        "all_confirmed_objects": {
            "states": dict(total("states")),
            "unknown_reasons": dict(total("reasons")),
            "ttc_status": dict(total("ttc_status")),
            "background_states": dict(background),
            "false_ttc_alarms": len(alarms),
        },
    }


def bag_list(paths):
    from eval_objects_gt import bag_dirs
    return [bag for path in paths for bag in bag_dirs(path)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--cache-dir")
    parser.add_argument("--json")
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-validity", action="store_true",
                        help="raw Kalman velocity, no validity checks")
    parser.add_argument("--per-bag", action="store_true")
    parser.add_argument("--virtual-plane", type=float, nargs=4,
                        metavar=("PX", "PY", "NX", "NY"),
                        help="hypothetical plane instead of the crane face")
    for name, default in [(f.name, f.default) for f in fields(MotionParams)]:
        parser.add_argument(f"--motion-{name.replace('_', '-')}",
                            type=type(default), default=default)
    for name, default in [(f.name, f.default) for f in fields(TtcParams)]:
        parser.add_argument(f"--ttc-{name.replace('_', '-')}",
                            type=type(default), default=default)
    args = parser.parse_args()

    motion_params = MotionParams(**{
        f.name: getattr(args, f"motion_{f.name}")
        for f in fields(MotionParams)})
    ttc_params = TtcParams(**{
        f.name: getattr(args, f"ttc_{f.name}") for f in fields(TtcParams)})

    runs, pooled, per_bag = [], defaultdict(list), {}
    for bag in bag_list(args.paths):
        data = detect_bag(bag, args.cache_dir)
        boundary = None
        if args.virtual_plane:
            px, py, nx, ny = args.virtual_plane
            boundary = PlaneBoundary((px, py), (nx, ny))
        result = run_bag(data, TrackerParams(), motion_params, ttc_params,
                         args.drop_rate, args.seed, not args.no_validity,
                         boundary)
        runs.append(result)
        summary = summarize(result)
        for key, values in summary.items():
            pooled[key].extend(values)
        name = os.path.basename(os.path.normpath(bag))
        if args.per_bag:
            per_bag[name] = report(summary, [result])
            print(name, json.dumps(per_bag[name]["static"]["states"]),
                  "alarms", len(result["alarms"]), flush=True)
    summary = report(pooled, runs)
    print(json.dumps(summary, indent=1))
    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump({"summary": summary, "per_bag": per_bag,
                       "motion_params": vars(motion_params),
                       "ttc_params": vars(ttc_params)}, handle)


if __name__ == "__main__":
    main()
