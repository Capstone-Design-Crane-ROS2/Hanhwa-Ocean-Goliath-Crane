#!/usr/bin/env python3
"""Describe the clusters that match no GT object, offline.

Same preprocessing, clustering and GT match as tools/eval_objects_gt.py
(one GT frame per scan). Every object that is not the match of a GT object
is recorded with its position in the eval (crane-face) frame, size, bottom
height, point count and the horizontal distance to the nearest GT box.
Positions are grouped on a 1 m grid (eval x, y) over the bag's frames.

Per grid cell (one candidate structure):
    frames_share   share of the bag's frames with an object in the cell
    near_gt        nearest GT box edge within --near-gt m (horizontal):
                   possibly a fragment of a GT object (split) rather than
                   background
    behind_face    eval x < 0 (crane side of the face)
    elevated       box bottom > --elevated m above the ground
    flat           box height < --flat m (ground remnant candidate)
Classes are descriptive, not a verdict: a persistent, GT-distant cell is
most likely a fixed structure; a flat one may be a ground-removal remnant;
a near-GT one may be a split piece of the GT object.

Usage:
    python3 tools/eval_unmatched_clusters.py <scenario-or-bag-dir>... \
        [--json out.json]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lidar"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_objects_gt import (  # noqa: E402
    EvalConfig,
    UnsupportedBag,
    bag_dirs,
    evaluate_frame,
    gt_boxes,
    load_preprocessed,
)


def footprint_distance(point, corners):
    """Horizontal distance from a point to a (yawed) GT box footprint."""
    origin = corners[0, :2]
    axes = [corners[1, :2] - origin, corners[3, :2] - origin]
    rel = point[:2] - origin
    outside = []
    for axis in axes:
        length = np.linalg.norm(axis)
        along = rel @ (axis / length)
        outside.append(max(0.0, -along, along - length))
    return float(np.hypot(*outside))


def analyze_bag(bag, args):
    pose, scene, frames = load_preprocessed(bag, gt_frames="scan")
    frame = scene["eval_frame"]
    ox, oy = float(frame["ox"]), float(frame["oy"])
    dx, sy = float(frame.get("dx", 1)), float(frame.get("sy", 1))
    sensor = pose.to_reference(np.zeros((1, 3)))[0]
    config = EvalConfig()

    cells = defaultdict(list)
    per_frame, matched_per_frame = [], []
    for k, item in enumerate(frames):
        boxes = gt_boxes(item["gt"], scene)
        _, _, objects, matches = evaluate_frame(
            item["points"], boxes, pose, config, sensor,
            return_objects=True)
        used = set(matches.values())
        unmatched = [o for o in objects if o.id not in used]
        per_frame.append(len(unmatched))
        matched_per_frame.append(len(used))
        for obj in unmatched:
            c = np.asarray(obj.center_reference)
            ex, ey = dx * (c[0] - ox), sy * (c[1] - oy)
            near = min((footprint_distance(c, corners)
                        for _, _, corners in boxes), default=np.inf)
            cells[(int(np.floor(ex)), int(np.floor(ey)))].append({
                "frame": k, "eval_xy": [ex, ey],
                "size": np.asarray(obj.size).tolist(),
                "bottom_z": float(obj.bbox_min[2]),
                "points": int(obj.point_count),
                "near_gt_m": float(near),
            })

    n = len(frames)
    summary_cells = []
    for key, items in cells.items():
        size = np.median([i["size"] for i in items], axis=0)
        bottom = float(np.median([i["bottom_z"] for i in items]))
        near = float(np.min([i["near_gt_m"] for i in items]))
        xy = np.median([i["eval_xy"] for i in items], axis=0)
        summary_cells.append({
            "cell": list(key),
            "eval_xy": [round(float(v), 2) for v in xy],
            "frames_share": round(len({i["frame"] for i in items}) / n, 3),
            "size_med": [round(float(v), 2) for v in size],
            "bottom_z_med": round(bottom, 2),
            "points_med": float(np.median([i["points"] for i in items])),
            "near_gt_min_m": round(near, 2),
            "near_gt": near <= args.near_gt,
            "behind_face": bool(xy[0] < 0),
            "elevated": bottom > args.elevated,
            "flat": float(size[2]) < args.flat,
        })
    summary_cells.sort(key=lambda c: -c["frames_share"])

    def share(pred):
        # Mean unmatched objects per frame falling in cells with pred.
        return round(sum(c["frames_share"] for c in summary_cells
                         if pred(c)), 2)
    return {
        "frames": n,
        "gt_selection": scene["eval_gt_selection"],
        "matched_per_frame": float(np.mean(matched_per_frame)) if n else 0,
        "unmatched_per_frame": float(np.mean(per_frame)) if n else 0,
        "unmatched_per_frame_by_kind": {
            "near_gt": share(lambda c: c["near_gt"]),
            "behind_face": share(lambda c: c["behind_face"]
                                 and not c["near_gt"]),
            "elevated": share(lambda c: c["elevated"] and not c["near_gt"]
                              and not c["behind_face"]),
            "flat": share(lambda c: c["flat"] and not c["near_gt"]),
            "persistent_other": share(
                lambda c: c["frames_share"] >= 0.8 and not (
                    c["near_gt"] or c["behind_face"] or c["elevated"]
                    or c["flat"])),
            "transient_other": share(
                lambda c: c["frames_share"] < 0.8 and not (
                    c["near_gt"] or c["behind_face"] or c["elevated"]
                    or c["flat"])),
        },
        "cells": summary_cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--json")
    parser.add_argument("--near-gt", type=float, default=1.0)
    parser.add_argument("--elevated", type=float, default=2.0)
    parser.add_argument("--flat", type=float, default=0.3)
    args = parser.parse_args()

    results = {}
    for path in args.paths:
        for bag in bag_dirs(path):
            name = os.path.basename(bag)
            try:
                result = analyze_bag(bag, args)
            except UnsupportedBag as exc:
                print(f"skipped: {exc}", flush=True)
                continue
            results[name] = result
            print(f"{name}: frames={result['frames']} "
                  f"matched/frame={result['matched_per_frame']:.2f} "
                  f"unmatched/frame={result['unmatched_per_frame']:.2f} "
                  f"{result['unmatched_per_frame_by_kind']}", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
