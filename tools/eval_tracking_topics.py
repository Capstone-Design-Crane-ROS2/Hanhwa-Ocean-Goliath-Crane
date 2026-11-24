#!/usr/bin/env python3
"""Check track IDs published during a bag replay against the bag's GT.

Integration counterpart of tools/eval_tracking_gt.py: instead of running the
modules offline it reads what the ROS node actually published, as recorded
by tools/object_topic_monitor.py (report "frames" = /lidar/objects JSON).

Each published frame is paired with the GT frame whose last LiDAR stamp
equals stamp_ns. A GT object is "detected" by the published object whose
center_reference lies inside its box (with MARGIN), the largest one if
several do. No point data is published, so this is coarser than the
point-based match of the offline evaluation.

Also checked: every object has a track_id field, confirmed IDs are unique
within a frame, and the frame-local id is still 0..n-1.

Usage:
    python3 tools/eval_tracking_topics.py <monitor-report.json> <bag-dir> \
        [--json out.json]
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from std_msgs.msg import String

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lidar"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_objects_gt import (  # noqa: E402
    GT_FRAME_MODES, gt_boxes, inside_box, select_gt_frames,
)
from eval_tracking_gt import summarize_object  # noqa: E402


MARGIN = 0.5


def read_gt(bag_dir, gt_frames="camera"):
    """({last LiDAR stamp: GT frame}, gt.json scene) of a bag.

    gt_frames "camera" (default, previous behavior, used by
    tools/eval_risk_gt.py): every MEASURING GT frame keyed by its last LiDAR
    stamp, a later frame of the same stamp replacing an earlier one.
    "scan": one GT frame per scan with boxes at the scan stamp
    (eval_objects_gt.select_gt_frames).
    """
    db = sqlite3.connect(
        f"file:{bag_dir}/bag/bag_0.db3?mode=ro&immutable=1", uri=True
    )
    topic = db.execute(
        "select id from topics where name='/eval/ground_truth'"
    ).fetchone()[0]
    gts = []
    for (data,) in db.execute(
        "select data from messages where topic_id=? order by timestamp",
        (topic,),
    ):
        gt = json.loads(deserialize_message(data, String).data)
        if (gt.get("measurement_state") == "MEASURING"
                and gt.get("lidar_stamps_ns")):
            gts.append(gt)
    with open(Path(bag_dir) / "gt.json", encoding="utf8") as handle:
        scene = json.load(handle)
    gts, _ = select_gt_frames(gts, scene, gt_frames)
    frames = {}
    for gt in gts:
        frames[gt["lidar_stamps_ns"][-1]] = gt
    return frames, scene


def check_frame(frame):
    """Structural problems of one published frame."""
    problems = []
    objects = frame["objects"]
    if any("track_id" not in obj for obj in objects):
        problems.append("missing_track_id")
    tids = [obj.get("track_id") for obj in objects
            if obj.get("track_id") is not None]
    if len(tids) != len(set(tids)):
        problems.append("duplicate_track_id")
    if [obj["id"] for obj in objects] != list(range(len(objects))):
        problems.append("frame_id_not_sequential")
    return problems


def evaluate(report, gt_frames, scene):
    entries = defaultdict(list)
    problems = defaultdict(int)
    paired = 0
    tracked_objects = 0
    total_objects = 0
    for index, frame in enumerate(report["frames"]):
        for problem in check_frame(frame):
            problems[problem] += 1
        total_objects += len(frame["objects"])
        tracked_objects += sum(o.get("track_id") is not None
                               for o in frame["objects"])
        gt = gt_frames.get(frame["stamp_ns"])
        if gt is None:
            continue
        paired += 1
        centers = np.array([o["center_reference"]
                            for o in frame["objects"]]).reshape(-1, 3)
        for gid, cls, corners in gt_boxes(gt, scene):
            inside = (inside_box(centers, corners, MARGIN)
                      if len(centers) else np.zeros(0, bool))
            candidates = np.flatnonzero(inside)
            obj = None
            if len(candidates):
                obj = max((frame["objects"][k] for k in candidates),
                          key=lambda o: o["point_count"])
            entries[gid].append({
                "index": index,
                "stamp_ns": frame["stamp_ns"],
                "class": cls,
                "status": "ok" if obj else "missed",
                "detected": obj is not None,
                "track_id": obj.get("track_id") if obj else None,
                "gt_center": corners.mean(axis=0).tolist(),
            })

    objects = {gid: summarize_object(items) for gid, items in entries.items()}
    for item in objects.values():
        item.pop("velocity_errors")
    totals = {key: sum(o[key] for o in objects.values())
              for key in ("detected", "tracked", "id_switches", "fragments")}
    return {
        "published_frames": len(report["frames"]),
        "frames_paired_with_gt": paired,
        "gt_measuring_frames": len(gt_frames),
        "objects_per_frame": total_objects / max(len(report["frames"]), 1),
        "tracked_share_of_objects": (
            tracked_objects / total_objects if total_objects else None
        ),
        "problems": dict(problems),
        **totals,
        "objects": objects,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("report")
    parser.add_argument("bag_dir")
    parser.add_argument("--json")
    parser.add_argument("--gt-frames", choices=GT_FRAME_MODES,
                        default="scan",
                        help="GT frames (default: one per LiDAR scan; "
                             "camera = previous behavior)")
    args = parser.parse_args()

    with open(args.report, encoding="utf8") as handle:
        report = json.load(handle)
    gt_frames, scene = read_gt(args.bag_dir, args.gt_frames)
    result = evaluate(report, gt_frames, scene)
    summary = {k: v for k, v in result.items() if k != "objects"}
    print(json.dumps(summary))
    for gid, item in result["objects"].items():
        print(f"  {gid}: detected={item['detected']} "
              f"tracked={item['tracked']} idsw={item['id_switches']} "
              f"frag={item['fragments']} ids={item['track_ids']}")
    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump(result, handle)


if __name__ == "__main__":
    main()
