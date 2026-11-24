#!/usr/bin/env python3
"""Offline check of lidar_merge_360: baseline code vs working copy.

Feeds the bag's /point_cloud_A and /rgb messages in recorded order straight
into LidarMerger callbacks (publishers replaced by lists, no DDS) and
compares:
    base       baseline code (git show <rev>:lidar/lidar_merge_360.py) with
               the dataset's params
    new_same   working-copy code, same params: must be identical to base
    new_lidar  one-scan bags (2nd dataset) only: camera_trigger:=false;
               checks every scan is published once with its own stamp and
               frame_id, merge_sync equals base, paired clouds unchanged

payload "synthetic" replaces the points by a few per-scan deterministic
points (reads only headers: fast, tests stamps / selection / counts);
"real" deserializes the real clouds (slow, also tests point data).

Usage:
    python3 tools/check_merge_offline.py <out.json> synthetic|real \
        [--baseline-rev HEAD] <bag_dir>...
Bags whose path contains "2차" are treated as one-scan bags.
"""
import hashlib
import importlib.util
import json
import sqlite3
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Header

REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = NEW = None


def header_of(data):
    sec, nsec, n = struct.unpack_from("<iII", data, 4)
    frame = data[16:16 + n - 1].decode()
    return sec * 10**9 + nsec, sec, nsec, frame


def read_bag(bag_dir, payload):
    db = sqlite3.connect(f"file:{bag_dir}/bag/bag_0.db3?mode=ro&immutable=1",
                         uri=True)
    topics = {n: i for i, n in db.execute("select id, name from topics")}
    ids = {topics["/point_cloud_A"]: "lidar", topics["/rgb"]: "camera"}
    out = []
    q = ("select id, topic_id, substr(data, 1, 128) from messages "
         "where topic_id in (?, ?) order by timestamp, id")
    for mid, tid, data in db.execute(q, tuple(ids)).fetchall():
        kind = ids[tid]
        ns, sec, nsec, frame = header_of(data)
        raw = None
        if kind == "lidar" and payload == "real":
            raw = db.execute("select data from messages where id=?",
                             (mid,)).fetchone()[0]
        out.append((kind, ns, sec, nsec, frame, raw))
    return out


def make_msg(mod, item):
    kind, ns, sec, nsec, frame, raw = item
    stamp = Time(sec=sec, nanosec=nsec)
    if kind == "camera":
        return Image(header=Header(stamp=stamp, frame_id=frame))
    if raw is not None:
        return deserialize_message(raw, PointCloud2)
    n = 1 + (ns // 1_000_000) % 7          # deterministic per scan
    pts = (np.arange(3 * n, dtype=np.float32).reshape(n, 3) + 1
           + (ns % 1000))
    return mod.make_cloud(pts, stamp, frame)


def run(mod, items, params):
    rclpy.init(args=["--ros-args", "--log-level", "fatal",
                     *sum((["-p", p] for p in params), [])])
    try:
        node = mod.LidarMerger()
        clouds, syncs = [], []
        node.cloud_pub.publish = lambda m: clouds.append((
            m.header.stamp.sec * 10**9 + m.header.stamp.nanosec,
            m.header.frame_id, m.width,
            hashlib.sha1(bytes(m.data)).hexdigest()))
        node.sync_pub.publish = lambda m: syncs.append(json.loads(m.data))
        for item in items:
            msg = make_msg(mod, item)
            if item[0] == "lidar":
                node.lidar_callback(msg)
            else:
                node.camera_callback(msg)
        node.report()
        counts = dict(node.counts)
        node.destroy_node()
    finally:
        rclpy.shutdown()
    return {"clouds": clouds, "syncs": syncs, "counts": counts}


def check_lidar_mode(items, res):
    scans = [i for i in items if i[0] == "lidar"]
    scan_stamps = [i[1] for i in scans]
    uniq_in = list(dict.fromkeys(scan_stamps))
    out = [c[0] for c in res["clouds"]]
    cams = sorted({i[1] for i in items if i[0] == "camera"})
    import bisect
    def has_cam(s):
        j = bisect.bisect_left(cams, s - 2_000_000)
        return j < len(cams) and abs(cams[j] - s) <= 2_000_000
    expected_sync = sum(1 for s in uniq_in if has_cam(s))
    frames_in = {i[1]: i[4] for i in scans}
    sync_pairs = [(s["camera_stamp_ns"], s["lidar_stamps_ns"]) for s in
                  res["syncs"]]
    return {
        "scans_in": len(scan_stamps),
        "scans_unique": len(uniq_in),
        "clouds_out": len(out),
        "out_equals_input_stamps": out == uniq_in,
        "out_duplicates": len(out) - len(set(out)),
        "out_monotonic": all(a < b for a, b in zip(out, out[1:])),
        "frame_id_kept": all(frames_in[c[0]] == c[1]
                             for c in res["clouds"]),
        "syncs": len(res["syncs"]),
        "syncs_expected": expected_sync,
        "sync_single_stamp": all(len(l) == 1 for _, l in sync_pairs),
        "sync_max_dt_ms": max((abs(c - l[0]) / 1e6 for c, l in sync_pairs),
                              default=0.0),
        "sync_camera_dup": len(sync_pairs) - len({c for c, _ in sync_pairs}),
        "sync_scan_dup": len(sync_pairs) - len({l[0] for _, l in sync_pairs}),
        "sync_scan_published": all(l[0] in set(out) for _, l in sync_pairs),
    }


def main():
    global BASE, NEW
    args = sys.argv[1:]
    rev = "HEAD"
    if "--baseline-rev" in args:
        k = args.index("--baseline-rev")
        rev = args[k + 1]
        del args[k:k + 2]
    out_path, payload, bags = args[0], args[1], args[2:]
    source = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{rev}:lidar/lidar_merge_360.py"],
        check=True, capture_output=True, text=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "lidar_merge_360_base.py"
        path.write_text(source)
        BASE = load("merge_base", path)
    NEW = load("merge_new", REPO / "lidar" / "lidar_merge_360.py")
    report = {"baseline_rev": rev, "payload": payload}
    for bag in bags:
        name = Path(bag).name
        items = read_bag(bag, payload)
        one_scan = "2차" in bag
        params = (["scan_offsets:=[0]", "log_skips:=false"] if one_scan
                  else ["scan_offsets:=[-1,0,1]", "log_skips:=true"])
        base = run(BASE, items, params)
        same = run(NEW, items, params)
        entry = {
            "base_clouds": len(base["clouds"]),
            "base_syncs": len(base["syncs"]),
            "base_counts": base["counts"],
            "new_same_identical": base == same,
        }
        if one_scan:
            lid = run(NEW, items, params + ["camera_trigger:=false"])
            entry["lidar_mode"] = check_lidar_mode(items, lid)
            entry["lidar_counts"] = lid["counts"]
            # Paired scans: same cloud content and sync as before.
            base_by = {c[0]: c for c in base["clouds"]}
            entry["paired_clouds_identical"] = all(
                c == base_by[c[0]] for c in lid["clouds"] if c[0] in base_by)
            entry["base_clouds_subset"] = set(base_by) <= {
                c[0] for c in lid["clouds"]}
            entry["syncs_identical_to_base"] = lid["syncs"] == base["syncs"]
        report[name] = entry
        print(name, json.dumps({k: v for k, v in entry.items()
                                if k not in ("base_counts", "lidar_counts")},
                               ensure_ascii=False), flush=True)
    Path(out_path).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
