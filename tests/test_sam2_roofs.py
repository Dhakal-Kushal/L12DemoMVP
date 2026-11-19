import unittest

import cv2
import numpy as np

from pipeline.sam2_roofs import mask_outline


class SamOutlineTests(unittest.TestCase):
    def test_rotated_mask_uses_original_image_coordinates(self):
        points = cv2.boxPoints(((130, 65), (50, 24), 32))
        mask = np.zeros((110, 220), np.uint8)
        cv2.fillConvexPoly(mask, points.round().astype(np.int32), 1)
        corners, reason = mask_outline(mask, [95, 40, 165, 90])
        self.assertIsNone(reason)
        # Corner ordering is arbitrary; each original corner should have a nearby match.
        distances = np.linalg.norm(points[:, None] - corners[None, :], axis=2)
        self.assertLess(distances.min(axis=1).max(), 2)

    def test_bad_masks_fall_back_to_unpadded_detector_box(self):
        mask = np.zeros((120, 120), np.uint8)
        prompt = [10, 20, 50, 60]
        expected = [[10, 20], [50, 20], [50, 60], [10, 60]]
        self.assertEqual(mask_outline(mask, prompt), (expected, "empty_mask"))
        mask[30, 15:35] = 1
        self.assertEqual(mask_outline(mask, prompt), (expected, "degenerate_mask"))
        mask[75:110, 70:100] = 1
        self.assertEqual(mask_outline(mask, prompt), (expected, "poor_prompt_overlap"))
