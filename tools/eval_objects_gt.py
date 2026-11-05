#!/usr/bin/env python3
"""Evaluate LiDAR object estimation against the dataset's GT, offline.

Reads a bag directly, applies the same functions as the ROS nodes (scan merge
by the GT's lidar_stamps_ns -> ROI -> voxel -> ground -> objects) and compares
every MEASURING frame with the GT boxes. Bags are only read.

Usage:
    python3 tools/eval_objects_gt.py <scenario-or-bag-dir>... \
        [--step N] [--json out.json] [--tolerance 0.6] ...
"""

import argparse
import glob
import json
import os
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lidar"))

from cluster_extractor import (  # noqa: E402
    ClusterParams,
    EuclideanClusterExtractor,
    label_components,
)
from ground_filter import GroundRemover, prior_from_transform  # noqa: E402
from lidar_merge_360 import extract_xyz  # noqa: E402
from object_estimator import (  # noqa: E402
    EstimatorParams,
    ReferencePose,
    estimate_objects,
    height_band_mask,
)
from roi_filter import filter_roi  # noqa: E402
from voxel_filter import voxel_downsample  # noqa: E402


# Minimum in-box points for a GT object to count as observable.
MIN_OBSERVED = 3
# Voxel centroids of an object can lie slightly outside its GT box.
BOX_MARGIN = 0.15
# A match needs this share of the box's points and of the cluster's points.
MIN_RECALL = 0.6
MIN_PURITY = 0.5


@dataclass(frozen=True)
class EvalConfig:
    cluster: ClusterParams = ClusterParams()
    estimator: EstimatorParams = EstimatorParams()
    crop_min_z: float = -1.0
    crop_max_z: float = 10.0


def load_preprocessed(bag_dir, step=1):
    """Preprocess a bag with the node functions.

    Returns:
        (pose, scene, frames); frames has one {"gt", "points"} dict per kept
        MEASURING frame.
    """
    db = sqlite3.connect(
        f"file:{bag_dir}/bag/bag_0.db3?mode=ro&immutable=1", uri=True
    )
    topics = {name: i for i, name in db.execute(
        "select id, name from topics"
    )}

    def messages(topic):
        return db.execute(
            "select data from messages where topic_id=? order by timestamp",
            (topics[topic],),
        )

    transform = None
    for (data,) in messages("/tf_static"):
        for item in deserialize_message(data, TFMessage).transforms:
            if item.child_frame_id == "OS1_A":
                transform = item.transform
    rotation = (transform.rotation.x, transform.rotation.y,
                transform.rotation.z, transform.rotation.w)
    translation = (transform.translation.x, transform.translation.y,
                   transform.translation.z)
    prior = prior_from_transform(rotation, translation, 0.0)

    gts = [
        json.loads(deserialize_message(data, String).data)
        for (data,) in messages("/eval/ground_truth")
    ]
    gts = [
        gt for gt in gts
        if gt.get("measurement_state") == "MEASURING"
        and gt.get("lidar_stamps_ns")
    ]

    scans = {}
    for (data,) in messages("/point_cloud_A"):
        msg = deserialize_message(data, PointCloud2)
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        scans[stamp] = msg

    remover = GroundRemover()
    frames = []
    for index, gt in enumerate(gts):
        # The ground tracker needs every frame; only stored frames are thinned.
        parts = [scans.get(stamp) for stamp in gt["lidar_stamps_ns"]]
        if any(part is None for part in parts):
            continue
        merged = np.concatenate([extract_xyz(part) for part in parts])
        roi, _ = filter_roi(merged)
        voxel = voxel_downsample(roi, 0.2)
        ground, _ = remover.process(voxel, prior)
        if index % step == 0:
            frames.append({"gt": gt, "points": voxel[~ground]})

    with open(os.path.join(bag_dir, "gt.json"), encoding="utf8") as handle:
        scene = json.load(handle)
    return ReferencePose(rotation, translation), scene, frames


def gt_boxes(frame_gt, scene):
    """(id, class, (8, 3) world corners) per GT object of a frame."""
    static = {
        item["object_id"]: item["box_corners_world"]
        for item in scene["ground_truth"]
    }
    return [
        (
            item["object_id"],
            item["object_class"],
            np.array(item.get("box_corners_world")
                     or static[item["object_id"]]),
        )
        for item in frame_gt["objects"]
    ]


def inside_box(world, corners, margin=BOX_MARGIN):
    """Points inside a possibly yawed box (bottom 4 then top 4 corners)."""
    origin = corners[0]
    inside = np.ones(len(world), dtype=bool)
    for corner in (corners[1], corners[3], corners[4]):
        axis = corner - origin
        length = np.linalg.norm(axis)
        along = (world - origin) @ (axis / length)
        inside &= (along > -margin) & (along < length + margin)
    return inside


def box_iou(min_a, max_a, min_b, max_b):
    overlap = np.clip(np.minimum(max_a, max_b) - np.maximum(min_a, min_b),
                      0, None).prod()
    union = (
        (max_a - min_a).prod() + (max_b - min_b).prod() - overlap
    )
    return float(overlap / union) if union > 0 else 0.0


def evaluate_frame(points, boxes, pose, config, sensor_world):
    """Run the module on a frame and compare it with the GT boxes.

    Returns:
        (records, stats): one record per GT object, and frame statistics.
    """
    start = time.perf_counter()
    keep = np.flatnonzero(height_band_mask(
        points, pose, config.crop_min_z, config.crop_max_z,
    ))
    cropped = points[keep]
    groups = EuclideanClusterExtractor(config.cluster).extract(cropped)
    cluster_ms = (time.perf_counter() - start) * 1e3

    start = time.perf_counter()
    objects, rejected = estimate_objects(
        cropped, groups, pose, config.estimator
    )
    estimate_ms = (time.perf_counter() - start) * 1e3

    world = pose.to_reference(cropped)
    labels = np.full(len(cropped), -1)
    for obj in objects:
        labels[obj.point_indices] = obj.id

    all_labels = None
    matched = set()
    records = []
    for object_id, cls, corners in boxes:
        mask = inside_box(world, corners)
        count = int(mask.sum())
        record = {
            "id": object_id, "class": cls, "observed_points": count,
            "gt_range": float(np.linalg.norm(
                corners.mean(axis=0) - sensor_world
            )),
        }
        records.append(record)
        if count < MIN_OBSERVED:
            record["status"] = "no_points"
            continue

        inside = labels[mask]
        inside = inside[inside >= 0]
        best = -1
        if len(inside):
            values, counts = np.unique(inside, return_counts=True)
            best = int(values[counts.argmax()])
            recall = counts.max() / count
            purity = counts.max() / objects[best].point_count

        if best >= 0 and purity < MIN_PURITY:
            record["status"] = "merged_kept"
            continue
        if best < 0 or recall < MIN_RECALL:
            # Absent or partial: was it dropped or split?
            if all_labels is None:
                all_labels = label_components(
                    cropped.astype(np.float64), config.cluster.tolerance
                )
            parts = all_labels[mask]
            values, counts = np.unique(parts, return_counts=True)
            biggest = values[counts.argmax()]
            size = int((all_labels == biggest).sum())
            if counts.max() / size < MIN_PURITY:
                record["status"] = "merged_dropped"
            elif best < 0:
                record["status"] = "too_small"
            else:
                record["status"] = "split"
            if best < 0 or record["status"] == "merged_dropped":
                continue

        obj = objects[best]
        matched.add(best)
        if record.get("status") is None:
            record["status"] = "ok"
        gt_min, gt_max = corners.min(axis=0), corners.max(axis=0)
        gt_center = (gt_min + gt_max) / 2.0
        centroid = pose.to_reference(obj.centroid[None])[0]
        bbox = (obj.bbox_min + obj.bbox_max) / 2.0
        record.update({
            "point_count": obj.point_count,
            "err_xy_bbox": float(np.linalg.norm((bbox - gt_center)[:2])),
            "err_xy_centroid": float(np.linalg.norm(
                (centroid - gt_center)[:2]
            )),
            "err_3d_bbox": float(np.linalg.norm(bbox - gt_center)),
            "err_3d_centroid": float(np.linalg.norm(centroid - gt_center)),
            "err_distance": obj.distance - float(np.linalg.norm(
                gt_center - sensor_world
            )),
            "err_horizontal_distance": obj.horizontal_distance - float(
                np.linalg.norm((gt_center - sensor_world)[:2])
            ),
            "size": obj.size.tolist(),
            "gt_size": (gt_max - gt_min).tolist(),
            "iou": box_iou(obj.bbox_min, obj.bbox_max, gt_min, gt_max),
        })

    stats = {
        "input_points": len(points),
        "cropped_points": len(cropped),
        "groups": len(groups),
        "rejected": rejected,
        "objects": len(objects),
        "unmatched": len(objects) - len(matched),
        "cluster_ms": cluster_ms,
        "estimate_ms": estimate_ms,
    }
    return records, stats


def evaluate_bag(bag_dir, config, step=1):
    pose, scene, frames = load_preprocessed(bag_dir, step)
    return evaluate_loaded(pose, scene, frames, config)


def evaluate_loaded(pose, scene, frames, config):
    records, stats = [], []
    sensor = pose.to_reference(np.zeros((1, 3)))[0]
    for frame in frames:
        frame_records, frame_stats = evaluate_frame(
            frame["points"], gt_boxes(frame["gt"], scene), pose, config,
            sensor,
        )
        records += frame_records
        stats.append(frame_stats)
    return records, stats


def summarize(records, stats):
    """Totals, statuses and error percentiles."""
    by_status = defaultdict(int)
    for record in records:
        by_status[record["status"]] += 1
    observable = len(records) - by_status["no_points"]
    summary = {
        "frames": len(stats),
        "gt_instances": len(records),
        "observable": observable,
        "status": dict(by_status),
        "ok_rate": (by_status["ok"] / observable) if observable else None,
    }
    good = [r for r in records if r["status"] == "ok"]
    for key in ("err_xy_bbox", "err_xy_centroid", "err_3d_bbox",
                "err_3d_centroid", "iou"):
        if good:
            values = [r[key] for r in good]
            summary[key] = {
                "p50": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
            }
    if good:
        for key in ("err_distance", "err_horizontal_distance"):
            values = np.array([r[key] for r in good])
            summary[key] = {
                "mean": float(values.mean()),
                "mae": float(np.abs(values).mean()),
                "p95_abs": float(np.percentile(np.abs(values), 95)),
            }
        size_error = np.array([
            np.subtract(r["size"], r["gt_size"]) for r in good
        ])
        summary["size_error_mean_xyz"] = size_error.mean(axis=0).tolist()
    if stats:
        for key in ("cluster_ms", "estimate_ms"):
            values = [s[key] for s in stats]
            summary[key] = {
                "p50": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "max": float(np.max(values)),
            }
        for key in ("cropped_points", "groups", "objects", "unmatched"):
            summary[key + "_per_frame"] = float(
                np.mean([s[key] for s in stats])
            )
    return summary


def bag_dirs(path):
    if os.path.isdir(os.path.join(path, "bag")):
        return [path]
    return sorted(
        item for item in glob.glob(os.path.join(path, "*"))
        if os.path.isdir(os.path.join(item, "bag"))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--json")
    parser.add_argument("--tolerance", type=float,
                        default=ClusterParams.tolerance)
    parser.add_argument("--min-points", type=int,
                        default=ClusterParams.min_points)
    parser.add_argument("--max-points", type=int,
                        default=ClusterParams.max_points)
    parser.add_argument("--max-footprint", type=float,
                        default=EstimatorParams.max_footprint)
    parser.add_argument("--crop-max-z", type=float,
                        default=EvalConfig.crop_max_z)
    args = parser.parse_args()

    config = EvalConfig(
        cluster=ClusterParams(
            args.tolerance, args.min_points, args.max_points,
        ),
        estimator=EstimatorParams(max_footprint=args.max_footprint),
        crop_max_z=args.crop_max_z,
    )

    results = {}
    for path in args.paths:
        for bag in bag_dirs(path):
            records, stats = evaluate_bag(bag, config, args.step)
            summary = summarize(records, stats)
            results[os.path.basename(bag)] = {
                "summary": summary, "records": records,
            }
            print(
                f"{os.path.basename(bag)}: "
                f"frames={summary['frames']} status={summary['status']} "
                f"objects/frame={summary.get('objects_per_frame', 0):.1f}"
            )

    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump(results, handle)


if __name__ == "__main__":
    main()
