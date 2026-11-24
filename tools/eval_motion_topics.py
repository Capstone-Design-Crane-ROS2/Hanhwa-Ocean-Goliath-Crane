#!/usr/bin/env python3
"""Check published velocity/TTC fields of a bag replay against the bag's GT.

Integration counterpart of tools/eval_motion_gt.py. Reads what the node
published (/lidar/objects JSON recorded by tools/object_topic_monitor.py)
and pairs each frame with the GT frame whose last LiDAR stamp equals
stamp_ns. A GT object is the published object whose center_reference lies
inside its box (as tools/eval_tracking_topics.py does).

Structure checks (every published object with motion fields):
    - motion and crane_boundary present whenever the frame has motion_model
    - velocity_mps / speed_mps null exactly when motion.valid is false
    - ttc_s and ttc_range_s null unless ttc_status is "valid"
    - statuses and states from the documented sets
    - predicted_tracks are not measured and their track IDs are not among
      the frame's detected track IDs
Accuracy (GT-matched objects):
    velocity error of moving GT (central difference of GT centers), false
    "moving" of static GT, crane-face distance error, TTC status counts.
Any "valid" TTC on this dataset is a false approach alarm: no GT object
approaches the crane face.

Usage:
    python3 tools/eval_motion_topics.py <monitor-report.json> <bag-dir> \
        [--json out.json]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lidar"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_objects_gt import gt_boxes, inside_box  # noqa: E402
from eval_tracking_topics import MARGIN, read_gt  # noqa: E402
from object_motion import MOTION_STATES, TTC_STATUSES  # noqa: E402


MOVING_SPEED = 0.3


def check_object(obj):
    """Structural problems of one published object (or predicted track)."""
    problems = []
    motion, risk = obj.get("motion"), obj.get("crane_boundary")
    if motion is None or risk is None:
        return ["missing_motion_fields"]
    if motion["state"] not in MOTION_STATES:
        problems.append("bad_state")
    if risk["ttc_status"] not in TTC_STATUSES:
        problems.append("bad_ttc_status")
    if motion["valid"] != (motion["state"] != "unknown"):
        problems.append("valid_flag_mismatch")
    if (motion["velocity_mps"] is None) == motion["valid"]:
        problems.append("velocity_null_mismatch")
    if (motion["speed_mps"] is None) == motion["valid"]:
        problems.append("speed_null_mismatch")
    is_valid = risk["ttc_status"] == "valid"
    if (risk["ttc_s"] is not None) != is_valid:
        problems.append("ttc_null_mismatch")
    if (risk["ttc_range_s"] is not None) != is_valid:
        problems.append("ttc_range_null_mismatch")
    if is_valid and not (risk["ttc_s"] > 0 and motion["state"] == "moving"):
        problems.append("ttc_without_moving")
    if obj.get("track_id", 0) is None and motion["reason"] != (
            "unconfirmed_track"):
        problems.append("untracked_with_motion")
    return problems


def evaluate(report, gt_frames, scene):
    problems = Counter()
    states, statuses, reasons = Counter(), Counter(), Counter()
    alarms = []
    entries = defaultdict(list)
    moving_ids = {g["object_id"] for g in scene["ground_truth"]
                  if g.get("moving")}
    frame = scene["eval_frame"]
    dx, ox = float(frame["dx"]), float(frame["ox"])
    motion_frames = paired = predicted = 0
    for index, data in enumerate(report["frames"]):
        if "motion_model" not in data:
            continue
        motion_frames += 1
        detected = {o.get("track_id") for o in data["objects"]}
        for obj in data["objects"]:
            for problem in check_object(obj):
                problems[problem] += 1
            if "motion" not in obj:
                continue
            states[obj["motion"]["state"]] += 1
            statuses[obj["crane_boundary"]["ttc_status"]] += 1
            if obj["motion"]["reason"]:
                reasons[obj["motion"]["reason"]] += 1
            if obj["crane_boundary"]["ttc_status"] == "valid":
                alarms.append({"stamp_ns": data["stamp_ns"],
                               "track_id": obj.get("track_id"),
                               **obj["crane_boundary"]})
        for item in data.get("predicted_tracks", []):
            predicted += 1
            for problem in check_object(item):
                problems["predicted_" + problem] += 1
            if item["motion"]["measured"]:
                problems["predicted_marked_measured"] += 1
            if item["track_id"] in detected:
                problems["predicted_also_detected"] += 1

        gt = gt_frames.get(data["stamp_ns"])
        if gt is None:
            continue
        paired += 1
        centers = np.array([o["center_reference"]
                            for o in data["objects"]]).reshape(-1, 3)
        for gid, cls, corners in gt_boxes(gt, scene):
            inside = (inside_box(centers, corners, MARGIN)
                      if len(centers) else np.zeros(0, bool))
            candidates = np.flatnonzero(inside)
            if not len(candidates):
                continue
            obj = max((data["objects"][k] for k in candidates),
                      key=lambda o: o["point_count"])
            if obj.get("track_id") is None or "motion" not in obj:
                continue
            entries[gid].append({
                "stamp_ns": data["stamp_ns"],
                "gt_center": corners.mean(axis=0)[:2],
                "gt_face_distance": float((dx * (corners[:, 0] - ox)).min()),
                "moving_gt": gid in moving_ids,
                "motion": obj["motion"],
                "risk": obj["crane_boundary"],
            })

    velocity_err, false_moving, static_frames = [], 0, 0
    face_err, moving_frames, moving_valid = [], 0, 0
    gt_status = Counter()
    for items in entries.values():
        for k, entry in enumerate(items):
            gt_status[entry["risk"]["ttc_status"]] += 1
            if entry["risk"]["distance_m"] is not None:
                face_err.append(entry["risk"]["distance_m"]
                                - entry["gt_face_distance"])
            gt_v = None
            if 0 < k < len(items) - 1:
                a, b = items[k - 1], items[k + 1]
                dt = (b["stamp_ns"] - a["stamp_ns"]) * 1e-9
                if dt > 0:
                    gt_v = (b["gt_center"] - a["gt_center"]) / dt
            if entry["moving_gt"] and gt_v is None:
                continue
            moving = (entry["moving_gt"]
                      and np.linalg.norm(gt_v) >= MOVING_SPEED)
            if moving:
                moving_frames += 1
                if entry["motion"]["valid"]:
                    moving_valid += 1
                    velocity_err.append(float(np.linalg.norm(
                        np.asarray(entry["motion"]["velocity_mps"]) - gt_v)))
            else:
                static_frames += 1
                false_moving += entry["motion"]["state"] == "moving"

    def stats(values):
        values = np.abs(np.asarray(values, dtype=np.float64))
        if not len(values):
            return {"n": 0}
        return {"n": int(len(values)), "mae": float(values.mean()),
                "p50": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(values.max())}

    return {
        "published_frames": len(report["frames"]),
        "frames_with_motion": motion_frames,
        "frames_paired_with_gt": paired,
        "problems": dict(problems),
        "states": dict(states),
        "unknown_reasons": dict(reasons),
        "ttc_status": dict(statuses),
        "predicted_tracks": predicted,
        "false_ttc_alarms": len(alarms),
        "alarms": alarms[:20],
        "gt_ttc_status": dict(gt_status),
        "moving_gt_frames": moving_frames,
        "moving_gt_valid_frames": moving_valid,
        "moving_velocity_error": stats(velocity_err),
        "static_gt_frames": static_frames,
        "static_false_moving": false_moving,
        "face_distance_error": stats(face_err),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report")
    parser.add_argument("bag_dir")
    parser.add_argument("--json")
    parser.add_argument("--gt-frames", choices=("camera", "scan"),
                        default="scan",
                        help="GT frames (default: one per LiDAR scan; "
                             "camera = previous behavior)")
    args = parser.parse_args()
    with open(args.report, encoding="utf8") as handle:
        report = json.load(handle)
    gt_frames, scene = read_gt(args.bag_dir, args.gt_frames)
    result = evaluate(report, gt_frames, scene)
    print(json.dumps({k: v for k, v in result.items() if k != "alarms"}))
    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump(result, handle)


if __name__ == "__main__":
    main()
