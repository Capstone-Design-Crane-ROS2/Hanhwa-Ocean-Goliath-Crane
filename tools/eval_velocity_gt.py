#!/usr/bin/env python3
"""Velocity accuracy of moving GT objects, offline (and published values).

Detection, tracking and motion validity are tools/eval_motion_gt.py
(detect_bag + run_bag with the default parameters): one GT frame per LiDAR
scan, boxes at the scan stamp. Only GT objects with motion in gt.json.

GT velocity, two references:
    central difference  (c[k+1] - c[k-1]) / (t[k+1] - t[k-1]) of the
                        per-scan GT box centers (real stamps, gaps included)
    path                gt.json motion: waypoints driven at speed_mps from
                        epoch_sim_s (straight segments); velocity is
                        speed * direction while moving, 0 before / after
The difference between the two shows where the central difference is not
reliable: frames whose difference window contains a start or a stop
("transition"). Accuracy uses the path velocity and leaves transition frames
out; both are reported.

Per frame of a moving GT object with a confirmed track:
    kalman velocity     tracker state (what motion validity checks)
    published velocity  kalman velocity if the motion state is not
                        "unknown", else null (what /lidar/objects carries)
    segment             first match wins:
                        before_start, stop_decay (<= --stop-s after the GT
                        stop), after_stop (GT stationary: is a speed
                        reported?), transition (central difference window
                        holds a start / stop), init (<= --init-s after the
                        track's first confirmed frame), after_start (<=
                        --start-s after the GT start: Kalman acceleration
                        response), gap (scan interval > --gap-ms in the last
                        --gap-hold-s), steady
Errors (vs the path velocity): vector |v - v_gt|, speed |v| - |v_gt|,
along-track and cross-track components, heading (|v_gt| >= 0.3 m/s).
Stats: n, MAE, RMSE, p50, p95, max.

Position lag (--lag): estimated box centers vs the path position at a
shifted time. For each object the time offset dt in [-0.3, 0.3] s that
minimizes the median |along-track error| over steady frames is reported,
with the along-track error at dt = 0 and the cross-track bias. A lag that is
the same in time for different speeds points to timing, the same in metres
to geometry (visible-surface bias).

Published values (--published-dir DIR with <bag>.objects.json from
tools/object_topic_monitor.py): the published object whose center is inside
the GT box (as tools/eval_tracking_topics.py) is compared at the same
stamp with the offline frame: velocity difference, motion state agreement.

Usage:
    python3 tools/eval_velocity_gt.py <bag-dir>... [--cache-dir DIR] \
        [--published-dir DIR] [--lag] [--json out.json]
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

from eval_motion_gt import bag_list, detect_bag, run_bag  # noqa: E402
from eval_objects_gt import UnsupportedBag, inside_box  # noqa: E402
from object_motion import MotionParams, TtcParams  # noqa: E402
from object_tracker import TrackerParams  # noqa: E402

MOVING_SPEED = 0.3


class PathMotion:
    """gt.json motion of one object in world xy (straight segments)."""

    def __init__(self, gt_item, eval_frame):
        motion = gt_item["motion"]
        ox, oy = float(eval_frame["ox"]), float(eval_frame["oy"])
        dx = float(eval_frame.get("dx", 1))
        sy = float(eval_frame.get("sy", 1))
        pts = np.array(motion["points"], dtype=float)
        self.points = np.column_stack([pts[:, 0] / dx + ox,
                                       pts[:, 1] / sy + oy])
        self.speed = float(motion["speed_mps"])
        self.epoch_ns = float(motion["epoch_sim_s"]) * 1e9
        seg = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(seg, axis=1)
        self.dirs = seg / self.lengths[:, None]
        self.cum = np.concatenate([[0.0], np.cumsum(self.lengths)])
        self.duration_ns = self.cum[-1] / self.speed * 1e9

    def _s(self, t_ns):
        return np.clip((t_ns - self.epoch_ns) * 1e-9 * self.speed,
                       0.0, self.cum[-1])

    def position(self, t_ns):
        s = self._s(t_ns)
        i = min(np.searchsorted(self.cum, s, side="right") - 1,
                len(self.dirs) - 1)
        return self.points[i] + self.dirs[i] * (s - self.cum[i])

    def velocity(self, t_ns):
        if not self.moving(t_ns):
            return np.zeros(2)
        s = self._s(t_ns)
        i = min(np.searchsorted(self.cum, s, side="right") - 1,
                len(self.dirs) - 1)
        return self.dirs[i] * self.speed

    def moving(self, t_ns):
        return 0.0 < t_ns - self.epoch_ns < self.duration_ns

    def direction(self, t_ns):
        s = self._s(t_ns)
        i = min(np.searchsorted(self.cum, s, side="right") - 1,
                len(self.dirs) - 1)
        return self.dirs[i]

    def since_start_s(self, t_ns):
        return (t_ns - self.epoch_ns) * 1e-9

    def since_stop_s(self, t_ns):
        return (t_ns - self.epoch_ns - self.duration_ns) * 1e-9

    def phase(self, t_ns):
        if t_ns <= self.epoch_ns:
            return "before_start"
        if t_ns >= self.epoch_ns + self.duration_ns:
            return "after_stop"
        return "moving"


def central_difference(stamps, centers):
    out = [None] * len(stamps)
    for k in range(1, len(stamps) - 1):
        dt = (stamps[k + 1] - stamps[k - 1]) * 1e-9
        if dt > 0:
            out[k] = (np.asarray(centers[k + 1]) - centers[k - 1]) / dt
    return out


def stats(values):
    v = np.asarray([x for x in values if x is not None], dtype=float)
    if not len(v):
        return {"n": 0}
    a = np.abs(v)
    return {"n": int(len(v)), "mae": round(float(a.mean()), 4),
            "rmse": round(float(np.sqrt((v ** 2).mean())), 4),
            "bias": round(float(v.mean()), 4),
            "p50": round(float(np.percentile(a, 50)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "max": round(float(a.max()), 4)}


def frame_rows(entries, path, args):
    """Per-frame analysis rows of one moving GT object (time order)."""
    stamps = [e["stamp_ns"] for e in entries]
    centers = [np.asarray(e["gt_center"]) for e in entries]
    cd = central_difference(stamps, centers)
    first_conf = {}
    rows, last_gap_ns = [], None
    for k, e in enumerate(entries):
        t = e["stamp_ns"]
        dt_prev = (t - stamps[k - 1]) * 1e-9 if k else None
        if dt_prev is not None and dt_prev * 1e3 > args.gap_ms:
            last_gap_ns = t
        v_path = path.velocity(t)
        row = {
            "index": e["index"], "stamp_ns": t, "dt_prev_s": dt_prev,
            "phase": path.phase(t),
            "v_path": v_path.tolist(),
            "v_cd": cd[k].tolist() if cd[k] is not None else None,
            "track_id": e.get("track_id"),
        }
        # Central difference window [k-1, k+1] contains a start / stop.
        if k and k + 1 < len(entries):
            row["transition"] = (path.phase(stamps[k - 1])
                                 != path.phase(stamps[k + 1]))
        else:
            row["transition"] = True
        if cd[k] is not None:
            row["cd_minus_path"] = float(np.linalg.norm(cd[k] - v_path))
        if "raw_velocity" in e and e["raw_velocity"] is not None:
            tid = e["track_id"]
            first_conf.setdefault(tid, t)
            v = np.asarray(e["raw_velocity"], dtype=float)
            published = None if e["state"] == "unknown" else v
            age = (t - first_conf[tid]) * 1e-9
            if row["phase"] == "before_start":
                segment = "before_start"
            elif row["phase"] == "after_stop":
                segment = ("stop_decay"
                           if path.since_stop_s(t) <= args.stop_s
                           else "after_stop")
            elif row["transition"]:
                segment = "transition"
            elif age <= args.init_s:
                segment = "init"
            elif path.since_start_s(t) <= args.start_s:
                segment = "after_start"
            elif last_gap_ns is not None and (
                    t - last_gap_ns) * 1e-9 <= args.gap_hold_s:
                segment = "gap"
            else:
                segment = "steady"
            d = path.direction(t)
            n = np.array([-d[1], d[0]])
            err = v - v_path
            row.update({
                "segment": segment, "track_age_s": age,
                "kalman": v.tolist(), "state": e["state"],
                "reason": e["reason"],
                "published": None if published is None else v.tolist(),
                "err_vector": float(np.linalg.norm(err)),
                "err_speed": float(np.linalg.norm(v) - np.linalg.norm(v_path)),
                "err_along": float(err @ d), "err_cross": float(err @ n),
                "heading_deg": (float(np.degrees(np.arctan2(
                    v[0] * d[1] - v[1] * d[0], v @ d)))
                    if np.linalg.norm(v_path) >= MOVING_SPEED else None),
                "est_center": np.asarray(e["center"]).tolist(),
                "gt_center": centers[k].tolist(),
            })
        rows.append(row)
    return rows


def lag_fit(rows, path):
    """Time offset of the estimated positions against the path."""
    steady = [r for r in rows if r.get("segment") == "steady"]
    if len(steady) < 10:
        return None
    t = np.array([r["stamp_ns"] for r in steady], dtype=float)
    est = np.array([r["est_center"] for r in steady])

    def along(shift_s):
        out = []
        for ti, ci in zip(t, est):
            p = path.position(ti - shift_s * 1e9)
            d = path.direction(ti)
            out.append((ci - p) @ d)
        return np.array(out)

    shifts = np.arange(-0.3, 0.3001, 0.005)
    cost = [np.median(np.abs(along(s))) for s in shifts]
    best = float(shifts[int(np.argmin(cost))])
    a0 = along(0.0)
    d0 = np.array([path.direction(ti) for ti in t])
    n0 = np.column_stack([-d0[:, 1], d0[:, 0]])
    p0 = np.array([path.position(ti) for ti in t])
    cross = np.einsum("ij,ij->i", est - p0, n0)
    gt_vs_path = np.array([np.linalg.norm(
        np.asarray(r["gt_center"]) - path.position(r["stamp_ns"]))
        for r in steady])
    return {
        "frames": len(steady),
        "along_err_at_0_mean_m": round(float(a0.mean()), 4),
        "along_err_at_0_median_m": round(float(np.median(a0)), 4),
        "cross_err_mean_m": round(float(cross.mean()), 4),
        "best_time_shift_s": round(best, 3),
        "best_shift_as_distance_m": round(best * path.speed, 4),
        "along_err_at_best_median_m": round(
            float(np.median(along(best))), 4),
        "gt_box_center_vs_path_median_m": round(
            float(np.median(gt_vs_path)), 4),
        "speed_mps": path.speed,
    }


def published_frames(path_json, gt_boxes_by_stamp):
    """{stamp: published object dict or None} for one GT object."""
    with open(path_json, encoding="utf8") as handle:
        report = json.load(handle)
    out = {}
    for frame in report["frames"]:
        box = gt_boxes_by_stamp.get(frame["stamp_ns"])
        if box is None:
            continue
        centers = np.array([o["center_reference"] for o in frame["objects"]]
                           ).reshape(-1, 3)
        inside = np.flatnonzero(inside_box(centers, box, 0.5)) \
            if len(centers) else []
        obj = max((frame["objects"][k] for k in inside),
                  key=lambda o: o["point_count"], default=None)
        out[frame["stamp_ns"]] = obj
    return out


def summarize_rows(rows):
    tracked = [r for r in rows if "kalman" in r]
    moving = [r for r in tracked if r["phase"] == "moving"]
    by_seg = defaultdict(list)
    for r in tracked:
        by_seg[r["segment"]].append(r)
    out = {
        "frames": len(rows), "tracked_frames": len(tracked),
        "segments": {s: len(v) for s, v in by_seg.items()},
        "gt_reference": {
            "cd_minus_path_all": stats([r.get("cd_minus_path")
                                        for r in rows]),
            "cd_minus_path_no_transition": stats([
                r.get("cd_minus_path") for r in rows
                if not r["transition"]]),
            "cd_minus_path_transition": stats([
                r.get("cd_minus_path") for r in rows if r["transition"]]),
        },
    }
    for name, sel in [("moving_all", moving),
                      ("moving_no_transition",
                       [r for r in moving if r["segment"] != "transition"]),
                      ("steady", by_seg.get("steady", [])),
                      ("init", by_seg.get("init", [])),
                      ("after_start", by_seg.get("after_start", [])),
                      ("gap", by_seg.get("gap", [])),
                      ("transition", by_seg.get("transition", []))]:
        pub = [r for r in sel if r["published"] is not None]
        out[name] = {
            "kalman_vector": stats([r["err_vector"] for r in sel]),
            "kalman_speed": stats([r["err_speed"] for r in sel]),
            "kalman_along": stats([r["err_along"] for r in sel]),
            "kalman_cross": stats([r["err_cross"] for r in sel]),
            "kalman_heading_deg": stats([r["heading_deg"] for r in sel]),
            "published_frames": len(pub),
            "published_share": round(len(pub) / len(sel), 3) if sel else None,
            "published_vector": stats([r["err_vector"] for r in pub]),
            "published_speed": stats([r["err_speed"] for r in pub]),
            "states": dict(Counter(r["state"] for r in sel)),
            "unknown_reasons": dict(Counter(
                r["reason"] for r in sel if r["state"] == "unknown")),
        }
    for name in ("before_start", "stop_decay", "after_stop"):
        sel = by_seg.get(name, [])
        out["gt_stationary_" + name] = {
            "frames": len(sel),
            "kalman_speed": stats([np.linalg.norm(r["kalman"])
                                   for r in sel]),
            "published_frames": sum(r["published"] is not None for r in sel),
            "published_moving": sum(r["state"] == "moving" for r in sel),
            "states": dict(Counter(r["state"] for r in sel)),
        }
    stationary = [r for r in tracked if r["phase"] != "moving"]
    out["gt_stationary"] = {
        "frames": len(stationary),
        "kalman_speed": stats([np.linalg.norm(r["kalman"])
                               for r in stationary]),
        "published_frames": sum(r["published"] is not None
                                for r in stationary),
        "states": dict(Counter(r["state"] for r in stationary)),
        "unknown_reasons": dict(Counter(
            r["reason"] for r in stationary if r["state"] == "unknown")),
    }
    # Frames at or above the p95 of the moving (no transition) vector error.
    sel = [r for r in moving if r["segment"] != "transition"]
    if sel:
        p95 = np.percentile([r["err_vector"] for r in sel], 95)
        out["p95_frames"] = [{
            k: r[k] for k in ("index", "stamp_ns", "segment", "track_age_s",
                              "dt_prev_s", "err_vector", "err_along",
                              "err_cross", "state", "reason")
        } for r in sel if r["err_vector"] >= p95]
    return out


def evaluate_bag(bag, args):
    data = detect_bag(bag, args.cache_dir, gt_frames="scan")
    scene = data["scene"]
    result = run_bag(data, TrackerParams(), MotionParams(), TtcParams())
    status = {}
    for k, frame in enumerate(data["frames"]):
        for rec in frame["records"]:
            status[(k, rec["id"])] = rec["status"]
    out = {"gt_selection": scene.get("eval_gt_selection"), "objects": {}}
    for item in scene["ground_truth"]:
        gid = item["object_id"]
        if not item.get("motion") or gid not in result["entries"]:
            continue
        path = PathMotion(item, scene["eval_frame"])
        entries = result["entries"][gid]
        rows = frame_rows(entries, path, args)
        for r in rows:
            r["cluster_status"] = status.get((r["index"], gid))
        obj = summarize_rows(rows)
        obj["cluster_status"] = dict(Counter(r["cluster_status"]
                                             for r in rows))
        if args.lag:
            obj["lag"] = lag_fit(rows, path)
        if args.published_dir:
            name = os.path.basename(os.path.normpath(bag))
            pub_path = os.path.join(args.published_dir,
                                    name + ".objects.json")
            if os.path.exists(pub_path):
                boxes = {}
                for frame in data["frames"]:
                    for b_gid, _, corners in frame["boxes"]:
                        if b_gid == gid:
                            boxes[frame["stamp_ns"]] = np.asarray(corners)
                pub = published_frames(pub_path, boxes)
                obj["published_compare"] = compare_published(rows, pub)
                obj["published_accuracy_all_frames"] = published_accuracy(
                    pub_path, path, args)
        if args.rows:
            obj["rows"] = rows
        out["objects"][gid] = obj
    return out


def published_accuracy(path_json, path, args, match_m=1.5):
    """Published velocity vs the path velocity over all published frames
    (also before the GT MEASURING phase), matched by path position."""
    with open(path_json, encoding="utf8") as handle:
        report = json.load(handle)
    rows = []
    for frame in report["frames"]:
        t = frame["stamp_ns"]
        p = path.position(t)
        best = None
        for obj in frame["objects"]:
            d = np.linalg.norm(np.asarray(obj["center_reference"][:2]) - p)
            if d <= match_m and (best is None or d < best[0]):
                best = (d, obj)
        if best is None or "motion" not in best[1]:
            continue
        m = best[1]["motion"]
        phase = path.phase(t)
        if phase == "before_start":
            seg = "before_start"
        elif phase == "after_stop":
            seg = ("stop_decay" if path.since_stop_s(t) <= args.stop_s
                   else "after_stop")
        elif path.since_start_s(t) <= args.start_s:
            seg = "after_start"
        else:
            seg = "moving"
        v = m["velocity_mps"]
        rows.append({"seg": seg, "state": m["state"], "reason": m["reason"],
                     "err": (float(np.linalg.norm(np.subtract(
                         v, path.velocity(t)))) if v is not None else None),
                     "speed": (float(np.linalg.norm(v))
                               if v is not None else None)})
    out = {}
    for seg in ("before_start", "after_start", "moving", "stop_decay",
                "after_stop"):
        sel = [r for r in rows if r["seg"] == seg]
        out[seg] = {
            "frames": len(sel),
            "published": sum(r["err"] is not None for r in sel),
            "published_vector_err": stats([r["err"] for r in sel]),
            "published_moving_state": sum(r["state"] == "moving"
                                          for r in sel),
            "unknown_reasons": dict(Counter(
                r["reason"] for r in sel if r["state"] == "unknown")),
        }
    return out


def compare_published(rows, pub):
    diffs, state_agree, n, offline_only, online_only = [], 0, 0, 0, 0
    online_null_moving, online_states = 0, Counter()
    for r in rows:
        if "kalman" not in r:
            continue
        obj = pub.get(r["stamp_ns"])
        if obj is None or "motion" not in obj:
            continue
        n += 1
        m = obj["motion"]
        online_states[m["state"]] += 1
        state_agree += m["state"] == r["state"]
        on = m["velocity_mps"]
        off = r["published"]
        if on is not None and off is not None:
            diffs.append(float(np.linalg.norm(np.subtract(on, off))))
        elif on is None and off is not None:
            offline_only += 1
        elif on is not None and off is None:
            online_only += 1
        if on is None and r["phase"] == "moving":
            online_null_moving += 1
    return {"paired_frames": n, "state_agreement": state_agree,
            "online_states": dict(online_states),
            "velocity_diff_both_published": stats(diffs),
            "published_offline_only": offline_only,
            "published_online_only": online_only,
            "online_null_while_gt_moving": online_null_moving}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--cache-dir")
    parser.add_argument("--published-dir")
    parser.add_argument("--json")
    parser.add_argument("--lag", action="store_true")
    parser.add_argument("--rows", action="store_true",
                        help="include per-frame rows in the JSON")
    parser.add_argument("--init-s", type=float, default=1.0)
    parser.add_argument("--start-s", type=float, default=1.0)
    parser.add_argument("--stop-s", type=float, default=1.5)
    parser.add_argument("--gap-ms", type=float, default=150.0)
    parser.add_argument("--gap-hold-s", type=float, default=0.5)
    args = parser.parse_args()

    results = {}
    for bag in bag_list(args.paths):
        name = os.path.basename(os.path.normpath(bag))
        try:
            results[name] = evaluate_bag(bag, args)
        except UnsupportedBag as exc:
            print(f"skipped: {exc}", flush=True)
            continue
        for gid, o in results[name]["objects"].items():
            s = o["moving_no_transition"]
            print(f"{name} {gid}: tracked={o['tracked_frames']} "
                  f"segments={o['segments']} "
                  f"kalman_vec={s['kalman_vector']} "
                  f"published_share={s['published_share']}", flush=True)
    if args.json:
        with open(args.json, "w", encoding="utf8") as handle:
            json.dump(results, handle, indent=1)


if __name__ == "__main__":
    main()
