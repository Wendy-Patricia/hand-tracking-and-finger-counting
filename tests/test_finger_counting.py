"""Tests de comptage avec des silhouettes générées, sans webcam ni modèle."""

import unittest
from itertools import combinations
from unittest.mock import patch

import cv2 as cv
import numpy as np

from hand_geometry import analyze_fingers
from temporal_filter import CountStabilizer
from config import ProcessingParameters
import camera


def hand_mask(count, arm_length=90, raised=None):
    mask = np.zeros((500, 440), np.uint8)
    cv.ellipse(mask, (220, 240), (64, 70), 0, 0, 360, 255, -1)
    cv.rectangle(mask, (186, 270), (254, 300 + arm_length), 255, -1)
    # Doigts arrondis de longueurs différentes et pouce latéral.
    fingers = [((188, 210), (173, 108), 12),
               ((216, 200), (212, 78), 12),
               ((244, 206), (252, 96), 12),
               ((268, 225), (286, 140), 10),
               ((173, 255), (121, 225), 13)]
    for index in (range(count) if raised is None else raised):
        base, tip, radius = fingers[index]
        cv.line(mask, base, tip, 255, 2 * radius)
        cv.circle(mask, tip, radius, 255, -1)
    return mask


def contour_of(mask, mode=cv.CHAIN_APPROX_SIMPLE):
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, mode)
    return max(contours, key=cv.contourArea)


class FingerCountingTests(unittest.TestCase):
    def test_all_finger_combinations_and_both_hands(self):
        for count in range(6):
            for raised in combinations(range(5), count):
                mask = hand_mask(count, raised=raised)
                for image in (mask, cv.flip(mask, 1)):
                    with self.subTest(raised=raised):
                        self.assertEqual(analyze_fingers(contour_of(image))['count'], count)

    def test_zero_to_five_at_different_scales_and_rotations(self):
        for count in range(6):
            for scale in (0.65, 1.0, 1.35):
                for angle in (-20, 0, 20):
                    with self.subTest(count=count, scale=scale, angle=angle):
                        mask = hand_mask(count)
                        rotation = cv.getRotationMatrix2D((220, 240), angle, 1)
                        mask = cv.warpAffine(mask, rotation, (440, 500), flags=cv.INTER_NEAREST)
                        mask = cv.resize(mask, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
                        analysis = analyze_fingers(contour_of(mask))
                        self.assertTrue(analysis['detected'])
                        self.assertEqual(analysis['count'], count, analysis)
                        self.assertTrue(analysis['reliable'])

    def test_lateral_thumb_alone_and_mirrored(self):
        mask = hand_mask(0)
        cv.line(mask, (173, 255), (121, 225), 255, 26)
        cv.circle(mask, (121, 225), 13, 255, -1)
        for image in (mask, cv.flip(mask, 1)):
            result = analyze_fingers(contour_of(image))
            self.assertEqual(result['count'], 1)
            self.assertIsNotNone(result['thumb'])

    def test_long_forearm_does_not_move_palm(self):
        for count in (0, 1, 5):
            short = analyze_fingers(contour_of(hand_mask(count, arm_length=20)))
            long = analyze_fingers(contour_of(hand_mask(count, arm_length=190)))
            self.assertEqual(long['count'], count)
            self.assertLessEqual(np.linalg.norm(np.subtract(short['center'], long['center'])), 2)

    def test_sparse_contour_and_translation_preserve_count(self):
        contour = contour_of(hand_mask(5))
        original = contour.copy()
        for shift in ([0, 0], [1200, 700], [-300, -400]):
            translated = contour + shift
            result = analyze_fingers(translated)
            self.assertEqual(result['count'], 5)
            for tip in result['fingertips']:
                self.assertGreaterEqual(cv.pointPolygonTest(translated.astype(np.int32), tip, True), -3)
        np.testing.assert_array_equal(contour, original)
        rectangle = np.array([[[0, 0]], [[80, 0]], [[80, 80]], [[0, 80]]])
        self.assertTrue(analyze_fingers(rectangle)['detected'])

    def test_contour_start_and_sampling_do_not_change_count(self):
        mask = hand_mask(5)
        for mode in (cv.CHAIN_APPROX_SIMPLE, cv.CHAIN_APPROX_NONE):
            contour = contour_of(mask, mode)
            for start in (0, len(contour) // 3):
                analysis = analyze_fingers(np.roll(contour, start, axis=0))
                self.assertEqual(analysis['count'], 5)
                self.assertEqual(len(analysis['fingertips']), analysis['count'])

    def test_empty_and_degenerate_contours(self):
        for contour in (None, np.empty((0, 1, 2)), np.array([[[1, 1]], [[2, 2]], [[3, 3]]])):
            self.assertFalse(analyze_fingers(contour)['detected'])


class TemporalTests(unittest.TestCase):
    def test_uncertain_frames_expire_old_count(self):
        stabilizer = CountStabilizer(window=5, min_votes=3)
        for _ in range(3):
            stabilizer.update(dict(detected=True, reliable=True, count=2))
        self.assertEqual(stabilizer.update(dict(detected=True, reliable=False)), 2)
        for _ in range(4):
            value = stabilizer.update(dict(detected=True, reliable=False))
        self.assertIsNone(value)

    def test_jitter_and_transition_to_fist(self):
        stabilizer = CountStabilizer(window=5, min_votes=3)
        for count in (5, 5, 4, 5):
            value = stabilizer.update(dict(detected=True, reliable=True, count=count))
        self.assertEqual(value, 5)
        for _ in range(5):
            value = stabilizer.update(dict(detected=True, reliable=True, count=0))
        self.assertEqual(value, 0)
        self.assertIsNone(stabilizer.update(dict(detected=False)))
        self.assertIsNone(stabilizer.stable_count)


class WebcamIntegrationTests(unittest.TestCase):
    def test_webcam_displays_stabilized_count(self):
        mask = hand_mask(5)
        hsv = np.zeros((*mask.shape, 3), dtype=np.uint8)
        hsv[mask > 0] = (10, 180, 210)
        frame = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        with patch.object(camera.cv, 'VideoCapture') as factory, \
                patch.object(camera.cv, 'namedWindow'), \
                patch.object(camera.cv, 'imshow'), \
                patch.object(camera.cv, 'waitKey', side_effect=[-1, -1, -1, ord('q')]), \
                patch.object(camera.cv, 'getWindowProperty', return_value=1), \
                patch.object(camera.cv, 'destroyAllWindows'), \
                patch.object(camera, 'create_trackbars'), \
                patch.object(camera, 'read_parameters', return_value=ProcessingParameters(
                    blur=0, erosions=0, dilations=0)), \
                patch.object(camera, 'draw_results', wraps=camera.draw_results) as draw:
            factory.return_value.isOpened.return_value = True
            factory.return_value.read.return_value = (True, frame)
            self.assertEqual(camera.main([]), 0)
            self.assertEqual(draw.call_count, 4)
            self.assertEqual(draw.call_args.args[2]['stable_count'], 5)
            factory.return_value.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
