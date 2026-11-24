#!/usr/bin/env python3
"""Evaluate frame-to-frame tracking against the GT object_id, offline.

Runs the same preprocessing and clustering as tools/eval_objects_gt.py on
every MEASURING frame, feeds the objects to ObjectTracker with the frame's
real stamp (last scan of lidar_stamps_ns, the node's header stamp), and
follows each GT object through the frames via the existing GT-cluster match.

Per GT object (only frames where it has points):
    detected   matched to a cluster (status ok or split)
    tracked    detected and its cluster has a confirmed track ID
    ID switch  tracked ID differs from the previous tracked ID of the object
    fragment   tracked -> not tracked -> tracked again
Totals also give MT/ML (tracked by its main ID in >= 80 % / < 20 % of its
detected frames) and, for moving GT, the track velocity error.

Events (--events, per GT object, for case analysis):
    confirmed   first frame with a confirmed track ID
    switch      tracked ID differs from the previous tracked ID
    lost        tracked -> observable but not tracked (status says whether
                the object was undetected, split, merged, or detected with
                a not yet confirmed track)
    reacquired  not tracked -> tracked again (a fragment)
Each event has the frame index (GT scan frame order), stamp_ns, track IDs,
the frame status and the statuses of the surrounding frames.

Robustness options:
    --drop-rate P   skip frames at random (tracker never sees them)
    --fixed-dt      give the tracker index * 83 ms instead of real stamps,
                    to show what a fixed period assumption would cost

Usage:
    python3 tools/eval_tracking_gt.py <scenario-or-bag-dir>... \
        [--json out.json] [--drop-rate 0.2] [--max-coast 1.5] ...
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lidar"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_objects_gt import (  # noqa: E402
    EvalConfig,
    bag_dirs,
    evaluate_frame,
    gt_boxes,
    load_preprocessed,
    GT_FRAME_MODES,
    UnsupportedBag,
)
from object_tracker import ObjectTracker, TrackerParams  # noqa: E402


DETECTED = ("ok", "split")
FIXED_PERIOD_NS = 83_333_333
MOSTLY_TRACKED = 0.8
MOSTLY_LOST = 0.2
# GT speed above this counts as moving, for the velocity error.
MOVING_SPEED = 0.3


def track_frames(pose, scene, frames, config, params, drop_rate=0.0,
                 seed=0, fixed_dt=False):
    """Runs detection + tracking over a bag's frames.

    Returns:
        (entries, stats): entries[gt_id] is a time-ordered list of per-frame
        dicts; stats has per-frame counts.
    """
    tracker = ObjectTracker(params)
    rng = np.random.default_rng(seed)
    sensor = pose.to_reference(np.zeros((1, 3)))[0]
    entries = defaultdict(list)
    stats = {"frames": 0, "dropped": 0, "confirmed_unmatched": [],
             "shared_id": 0}
    for index, frame in enumerate(frames):
        if drop_rate > 0 and rng.random() < drop_rate:
            stats["dropped"] += 1
            continue
        boxes = gt_boxes(frame["gt"], scene)
        records, _, objects, matches = evaluate_frame(
            frame["points"], boxes, pose, config, sensor,
            return_objects=True,
        )
        stamp = (index * FIXED_PERIOD_NS if fixed_dt
                 else frame["gt"]["lidar_stamps_ns"][-1])
        real_stamp = frame["gt"]["lidar_stamps_ns"][-1]
        ids = tracker.update(
            stamp, [obj.center_reference for obj in objects], "world",
        )
        stats["frames"] += 1

        centers = {gid: corners.mean(axis=0) for gid, _, corners in boxes}
        frame_tids = []
        for record in records:
            gid = record["id"]
            if record["status"] == "no_points":
                continue
            obj_index = matches.get(gid)
            tid = ids[obj_index] if obj_index is not None else None
            entry = {
                "index": index,
                "stamp_ns": real_stamp,
                "class": record["class"],
                "status": record["status"],
                "detected": obj_index is not None,
                "track_id": tid,
                "gt_center": centers[gid].tolist(),
            }
            if tid is not None:
                track = tracker.tracks[tid]
                entry["track_velocity"] = track.velocity.tolist()
                entry["track_hits"] = track.hits
                frame_tids.append(tid)
            entries[gid].append(entry)

        counts = Counter(frame_tids)
        stats["shared_id"] += sum(c - 1 for c in counts.values() if c > 1)
        confirmed = [t for t in ids if t is not None]
        stats["confirmed_unmatched"].append(
            len(set(confirmed) - set(frame_tids))
        )
    stats["resets"] = tracker.resets
    stats["tracks_created"] = tracker.next_id
    return entries, stats


def gt_velocities(entries):
    """GT velocity per entry from neighbouring GT centers (central diff)."""
    out = [None] * len(entries)
    for k in range(1, len(entries) - 1):
        a, b = entries[k - 1], entries[k + 1]
        dt = (b["stamp_ns"] - a["stamp_ns"]) * 1e-9
        if dt > 0:
            out[k] = (np.subtract(b["gt_center"], a["gt_center"]) / dt)
    return out


def summarize_object(entries):
    detected = [e for e in entries if e["detected"]]
    tracked = [e for e in entries if e["track_id"] is not None]
    switches, fragments = 0, 0
    last_id, was_tracked, ever_tracked = None, False, False
    for entry in entries:
        tid = entry["track_id"]
        if tid is not None:
            if last_id is not None and tid != last_id:
                switches += 1
            if ever_tracked and not was_tracked:
                fragments += 1
            last_id, was_tracked, ever_tracked = tid, True, True
        else:
            was_tracked = False
    ids = Counter(e["track_id"] for e in tracked)
    main_share = (ids.most_common(1)[0][1] / len(detected)
                  if detected and ids else 0.0)
    first_detected = next((k for k, e in enumerate(entries)
                           if e["detected"]), None)
    first_tracked = next((k for k, e in enumerate(entries)
                          if e["track_id"] is not None), None)

    velocity_errors = []
    for entry, gt_v in zip(entries, gt_velocities(entries)):
        if gt_v is None or "track_velocity" not in entry:
            continue
        if np.linalg.norm(gt_v[:2]) < MOVING_SPEED:
            continue
        velocity_errors.append(float(np.linalg.norm(
            np.subtract(entry["track_velocity"][:2], gt_v[:2])
        )))

    return {
        "class": entries[0]["class"],
        "observable": len(entries),
        "detected": len(detected),
        "tracked": len(tracked),
        "id_switches": switches,
        "fragments": fragments,
        "track_ids": sorted(ids),
        "main_id_share": main_share,
        "init_delay_frames": (
            first_tracked - first_detected
            if first_tracked is not None and first_detected is not None
            else None
        ),
        "velocity_errors": velocity_errors,
    }


def object_events(entries, context=2):
    """ID events of one GT object (see module docstring)."""
    events = []
    last_id, was_tracked = None, False
    for k, entry in enumerate(entries):
        tid = entry["track_id"]
        event = None
        if tid is not None and last_id is None:
            event = {"type": "confirmed", "track_id": tid}
        elif tid is not None and tid != last_id:
            event = {"type": "switch", "from": last_id, "to": tid}
        elif tid is not None and not was_tracked:
            event = {"type": "reacquired", "track_id": tid}
        elif tid is None and was_tracked:
            event = {"type": "lost", "track_id": last_id}
        if event is not None:
            event.update({
                "index": entry["index"],
                "stamp_ns": entry["stamp_ns"],
                "status": entry["status"],
                "detected": entry["detected"],
                "gap_frames": (entry["index"] - entries[k - 1]["index"] - 1
                               if k else 0),
                "window": [(e["index"], e["status"], e["track_id"])
                           for e in entries[max(0, k - context):
                                            k + context]],
            })
            events.append(event)
        if tid is not None:
            last_id = tid
        was_tracked = tid is not None
    return events


def summarize(objects, stats):
    totals = {key: sum(o[key] for o in objects.values())
              for key in ("observable", "detected", "tracked",
                          "id_switches", "fragments")}
    with_detection = [o for o in objects.values() if o["detected"]]
    velocity = [v for o in objects.values() for v in o["velocity_errors"]]
    delays = [o["init_delay_frames"] for o in objects.values()
              if o["init_delay_frames"] is not None]
    summary = {
        "frames": stats["frames"],
        "dropped_frames": stats["dropped"],
        "gt_objects": len(objects),
        **totals,
        "tracked_of_detected": (
            totals["tracked"] / totals["detected"]
            if totals["detected"] else None
        ),
        "mostly_tracked": sum(o["main_id_share"] >= MOSTLY_TRACKED
                              for o in with_detection),
        "mostly_lost": sum(o["main_id_share"] < MOSTLY_LOST
                           for o in with_detection),
        "shared_id": stats["shared_id"],
        "resets": stats["resets"],
        "tracks_created": stats["tracks_created"],
        "confirmed_unmatched_per_frame": float(
            np.mean(stats["confirmed_unmatched"])
        ) if stats["confirmed_unmatched"] else 0.0,
        "init_delay_frames_max": max(delays, default=None),
    }
    if velocity:
        summary["velocity_error_xy"] = {
            "n": len(velocity),
            "p50": float(np.median(velocity)),
            "p95": float(np.percentile(velocity, 95)),
        }
    return summary


def evaluate_loaded(pose, scene, frames, config, params, events=False,
                    **options):
    """(summary, per-object results) of preprocessed frames."""
    entries, stats = track_frames(pose, scene, frames, config, params,
                                  **options)
    objects = {gid: summarize_object(items)
               for gid, items in entries.items()}
    if events:
        for gid, items in entries.items():
            objects[gid]["events"] = object_events(items)
    return summarize(objects, stats), objects


def evaluate_bag(bag_dir, config, params, events=False, gt_frames="camera",
                 **options):
    """(summary, per-object results) of one bag.

    gt_frames: "camera" (every GT frame, the previous behavior and the
    library default) or "scan" (one per LiDAR scan, eval_objects_gt).
    """
    pose, scene, frames = load_preprocessed(bag_dir, gt_frames=gt_frames)
    return evaluate_loaded(pose, scene, frames, config, params, events,
                           **options)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json")
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fixed-dt", action="store_true")
    parser.add_argument("--gt-frames", choices=GT_FRAME_MODES,
                        default="scan",
                        help="GT frames (default: one per LiDAR scan; "
                             "camera = every GT frame, previous behavior)")
    parser.add_argument("--events", action="store_true",
                        help="add per-object ID events to the JSON")
    defaults = TrackerParams()
    parser.add_argument("--min-hits", type=int, default=defaults.min_hits)
    parser.add_argument("--max-coast", type=float,
                        default=defaults.max_coast_s)
    parser.add_argument("--accel-std", type=float,
                        default=defaults.accel_std)
    parser.add_argument("--meas-std-xy", type=float,
                        default=defaults.meas_std_xy)
    parser.add_argument("--gate-distance", type=float,
                        default=defaults.max_gate_distance)
    args = parser.parse_args()

    params = TrackerParams(
        min_hits=args.min_hits, max_coast_s=args.max_coast,
        accel_std=args.accel_std, meas_std_xy=args.meas_std_xy,
        max_gate_distance=args.gate_distance,
    )
    results = {}
    for path in args.paths:
        for bag in bag_dirs(path):
            try:
                pose, scene, frames = load_preprocessed(
                    bag, gt_frames=args.gt_frames)
                selection = scene.get("eval_gt_selection")
                summary, objects = evaluate_loaded(
                    pose, scene, frames, EvalConfig(), params,
                    events=args.events,
                    drop_rate=args.drop_rate, seed=args.seed,
                    fixed_dt=args.fixed_dt,
                )
            except UnsupportedBag as exc:
                print(f"skipped: {exc}", flush=True)
                continue
            name = os.path.basename(bag)
            results[name] = {"summary": summary, "objects": objects}
            if selection and (selection["duplicates_removed"]
                              or selection["objects_shifted_to_lidar"]):
                # 2nd dataset only; 1st dataset output is unchanged.
                results[name]["gt_selection"] = selection
                print(f"{name}: GT {selection}")
            print(
                f"{name}: frames={summary['frames']} "
                f"gt={summary['gt_objects']} "
                f"tracked/detected={summary['tracked']}/"
                f"{summary['detected']} idsw={summary['id_switches']} "
                f"frag={summary['fragments']} "
                f"MT={summary['mostly_tracked']} ML={summary['mostly_lost']}"
                + (f" vel_err_p50="
                   f"{summary['velocity_error_xy']['p50']:.2f}m/s"
                   if "velocity_error_xy" in summary else ""),
                flush=True,
            )
    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump(results, handle)


if __name__ == "__main__":
    main()
