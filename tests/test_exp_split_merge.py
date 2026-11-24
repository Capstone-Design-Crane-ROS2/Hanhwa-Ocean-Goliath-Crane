"""Experimental fragment handling of tools/exp_split_merge.py."""

import sys
import unittest
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "lidar"))
sys.path.insert(0, str(root / "tools"))

try:
    from exp_split_merge import Grouper, merge_fragments, standing_column
except ImportError:  # pragma: no cover - ROS not sourced
    merge_fragments = None


def block(lo, hi, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, (n, 3))


@unittest.skipIf(merge_fragments is None, "ROS 2 Python not available")
class MergeTest(unittest.TestCase):
    def run_merge(self, parts, ratio=0.3, contained=0.8):
        world = np.vstack(parts)
        groups, start = [], 0
        for p in parts:
            groups.append(np.arange(start, start + len(p)))
            start += len(p)
        merged, n = merge_fragments(groups, world, ratio, contained, 1.2)
        return [sorted(set(np.searchsorted(np.cumsum([len(p) for p in parts]),
                                           g, side="right")))
                for g in merged], n

    def test_roof_piece_joins_body(self):
        body = block([0, 0, 0.3], [8, 2.5, 2.2], 400)
        roof = block([2, 0.8, 2.9], [5, 1.7, 3.1], 25, seed=1)
        groups, n = self.run_merge([body, roof])
        self.assertEqual(n, 1)
        self.assertEqual(groups, [[0, 1]])

    def test_person_column_inside_box_is_never_merged(self):
        body = block([0, 0, 0.3], [8, 2.5, 2.2], 400)
        person = block([3.8, 2.6, 0.0], [4.2, 2.75, 1.7], 40, seed=2)
        self.assertTrue(standing_column(person))
        groups, n = self.run_merge([body, person])
        self.assertEqual(n, 0)
        self.assertEqual(len(groups), 2)

    def test_large_or_far_or_outside_piece_kept(self):
        body = block([0, 0, 0.3], [8, 2.5, 2.2], 400)
        half = block([8.7, 0, 0.3], [12, 2.5, 2.2], 200, seed=3)  # outside
        far_roof = block([2, 0.8, 4.0], [5, 1.7, 4.2], 25, seed=4)  # gap 1.8
        big_inside = block([1, 0.5, 2.5], [6, 2.0, 3.0], 200, seed=5)
        for piece in (half, far_roof, big_inside):
            _, n = self.run_merge([body, piece])
            self.assertEqual(n, 0)

    def test_no_chain(self):
        body = block([0, 0, 0.3], [8, 2.5, 2.2], 400)
        a = block([2, 0.8, 2.9], [5, 1.7, 3.1], 25, seed=6)
        # b sits on a (inside a's box) but would only reach body through a.
        b = block([3, 1.0, 3.6], [3.5, 1.4, 3.7], 5, seed=7)
        groups, n = self.run_merge([body, a, b])
        self.assertEqual(n, 1)
        self.assertIn([2], groups)


@unittest.skipIf(merge_fragments is None, "ROS 2 Python not available")
class HullTest(unittest.TestCase):
    def test_empty_box_corner_of_rotated_body(self):
        # Body rotated 45 deg: a diamond; a cone in the empty box corner.
        rng = np.random.default_rng(0)
        u = rng.uniform(-1, 1, (400, 2)) * [4.0, 1.2]
        rot = np.array([[1, -1], [1, 1]]) / np.sqrt(2)
        xy = u @ rot.T
        body = np.column_stack([xy, rng.uniform(0.3, 2.2, 400)])
        lo = body.min(axis=0)
        # Box corner region, 0.7-1.1 m outside the diamond's edge.
        cone = block([lo[0] + 0.1, lo[1] + 0.1, 0.0],
                     [lo[0] + 0.35, lo[1] + 0.35, 0.7], 20, seed=8)
        world = np.vstack([body, cone])
        groups = [np.arange(400), np.arange(400, 420)]
        _, n_box = merge_fragments(groups, world, 0.5, 0.75, 5.0)
        _, n_hull = merge_fragments(groups, world, 0.5, 0.75, 5.0,
                                    hull=True)
        self.assertEqual(n_box, 1)     # inside the axis-aligned box
        self.assertEqual(n_hull, 0)    # outside the real footprint


@unittest.skipIf(merge_fragments is None, "ROS 2 Python not available")
class GrouperTest(unittest.TestCase):
    def test_link_after_persistence(self):
        body = block([0, 0, 0.3], [8, 2.5, 2.2], 400)
        roof = block([2, 0.8, 2.9], [5, 1.7, 3.1], 25, seed=1)
        g = Grouper(0.5, 0.75, 1.2, persist=3)
        out = [g.update([7, 9], [body, roof]) for _ in range(4)]
        self.assertEqual(out[0], [7, 9])
        self.assertEqual(out[1], [7, 9])
        self.assertEqual(out[2], [7, 7])
        self.assertEqual(out[3], [7, 7])
        # Relation gone: the fragment track is reported under its own ID.
        far = roof + np.array([0, 0, 5.0])
        self.assertEqual(g.update([7, 9], [body, far]), [7, 9])


if __name__ == "__main__":
    unittest.main()
