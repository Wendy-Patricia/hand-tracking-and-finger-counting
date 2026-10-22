"""Tests sans webcam : couleurs, contours et cycle de vie de l'application."""

import contextlib
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import cv2 as cv
import numpy as np

import camera
from config import ProcessingParameters
from hand_detection import contour_from_mask, describe_hand, detect_hand
from finger_counting import count_fingers
from pipeline import process_frame
from rendering import draw_detection
from segmentation import segment_hand


class ProcessingTests(unittest.TestCase):
    def test_hue_wrap_and_reversed_saturation_and_value_bounds(self):
        hsv = np.array([[[175, 200, 200], [5, 200, 200], [90, 200, 200],
                         [5, 0, 200], [5, 200, 20]]], dtype=np.uint8)
        frame = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        parameters = ProcessingParameters(
            h_min=170, h_max=15, s_min=255, s_max=100,
            v_min=255, v_max=100, blur=0, erosions=0, dilations=0,
        )
        _, mask = segment_hand(frame, parameters)
        np.testing.assert_array_equal(mask, [[255, 255, 0, 0, 0]])

    def test_largest_contour_and_minimum_area_boundary(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv.rectangle(mask, (5, 5), (15, 15), 255, -1)
        cv.rectangle(mask, (30, 30), (70, 70), 255, -1)
        contour = contour_from_mask(mask, min_area=1600)
        detection = describe_hand(contour)
        self.assertIsNotNone(detection)
        self.assertEqual(detection.bounding_box, (30, 30, 41, 41))
        self.assertEqual(detection.area, 1600)
        self.assertEqual(detection.solidity, 1.0)
        self.assertEqual(detection.shape, "Silhouette compacte")
        self.assertIsNone(contour_from_mask(mask, min_area=1601))

    def test_empty_mask_has_no_detection(self):
        self.assertIsNone(detect_hand(np.zeros((100, 100, 3), dtype=np.uint8)))

    def test_public_contract_accepts_bgr_and_returns_contour(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        frame[50:150, 50:150] = cv.cvtColor(
            np.uint8([[[10, 200, 200]]]), cv.COLOR_HSV2BGR
        )[0, 0]
        original = frame.copy()
        contour = detect_hand(frame)
        self.assertIsInstance(contour, np.ndarray)
        self.assertEqual(contour.shape[1:], (1, 2))
        count, fingertips = count_fingers(contour)
        self.assertEqual(count, 0)
        self.assertEqual(fingertips, [])
        np.testing.assert_array_equal(frame, original)

    def test_finger_count_handles_missing_and_degenerate_contours(self):
        for contour in (None, np.empty((0, 1, 2), dtype=np.int32),
                        np.array([[[0, 0]], [[10, 10]], [[20, 20]]], dtype=np.int32)):
            self.assertEqual(count_fingers(contour), (0, []))

    def test_finger_tips_use_original_frame_coordinates(self):
        contour = np.array([
            [60, 160], [60, 80], [80, 80], [80, 20],
            [90, 10], [100, 20], [100, 80], [120, 80], [120, 160],
        ], dtype=np.int32).reshape(-1, 1, 2)
        original = contour.copy()
        self.assertEqual(count_fingers(contour), (1, [(90, 10)]))
        offset = np.array([[[150, 100]]], dtype=np.int32)
        self.assertEqual(count_fingers(contour + offset), (1, [(240, 110)]))
        self.assertEqual(count_fingers(contour[::-1].copy()), (1, [(90, 10)]))
        np.testing.assert_array_equal(contour, original)

    def test_separated_fingers_of_different_lengths(self):
        contour = np.array([
            [40, 180], [40, 90], [45, 90], [45, 35], [53, 25],
            [61, 35], [61, 90], [80, 90], [80, 20], [88, 10],
            [96, 20], [96, 90], [115, 90], [115, 45], [123, 35],
            [131, 45], [131, 90], [140, 90], [140, 180],
        ], dtype=np.int32).reshape(-1, 1, 2)
        self.assertEqual(count_fingers(contour), (3, [(53, 25), (88, 10), (123, 35)]))

    def test_pipeline_processes_image_without_windows_or_source_mutation(self):
        hsv = np.zeros((120, 160, 3), dtype=np.uint8)
        hsv[30:90, 40:120] = (10, 200, 200)
        frame = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        original = frame.copy()
        with patch.object(cv, "namedWindow", side_effect=AssertionError("GUI utilisée")):
            result = process_frame(frame, ProcessingParameters())
            self.assertIsNotNone(result.detection)
            filtered = process_frame(frame, replace(ProcessingParameters(), min_area=30000))
        self.assertIsNone(filtered.detection)
        self.assertEqual((filtered.count, filtered.fingertips), (0, []))
        self.assertIsNone(filtered.contour)
        self.assertIs(result.contour, result.detection.contour)
        self.assertEqual((result.count, result.fingertips), count_fingers(result.contour))
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(result.mask.shape, frame.shape[:2])
        self.assertEqual(set(np.unique(result.mask)), {0, 255})
        display = draw_detection(frame.copy(), result.detection)
        self.assertFalse(np.array_equal(display, original))
        np.testing.assert_array_equal(frame, original)


class CameraLifecycleTests(unittest.TestCase):
    def test_unavailable_camera_releases_resources(self):
        with patch.object(camera.cv, "VideoCapture") as factory, \
                patch.object(camera.cv, "destroyAllWindows") as close, \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            factory.return_value.isOpened.return_value = False
            self.assertEqual(camera.main(["--camera", "2"]), 1)
        factory.assert_called_once_with(2)
        factory.return_value.release.assert_called_once_with()
        close.assert_called_once_with()
        self.assertIn("Impossible d'ouvrir la webcam", errors.getvalue())

    def test_quit_after_processing_frame_releases_resources(self):
        with patch.object(camera.cv, "VideoCapture") as factory, \
                patch.object(camera.cv, "namedWindow"), \
                patch.object(camera.cv, "imshow"), \
                patch.object(camera.cv, "waitKey", return_value=ord("q")), \
                patch.object(camera.cv, "destroyAllWindows") as close, \
                patch.object(camera, "create_trackbars"), \
                patch.object(camera, "read_parameters", return_value=ProcessingParameters()), \
                contextlib.redirect_stdout(io.StringIO()):
            factory.return_value.isOpened.return_value = True
            factory.return_value.read.return_value = (
                True, np.zeros((120, 160, 3), dtype=np.uint8)
            )
            self.assertEqual(camera.main([]), 0)
        factory.return_value.release.assert_called_once_with()
        close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
