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
from hand_detection import detect_hand
from pipeline import process_frame
from visualization import draw_detection
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
        detection = detect_hand(mask, min_area=1600)
        self.assertIsNotNone(detection)
        self.assertEqual(detection.bounding_box, (30, 30, 41, 41))
        self.assertEqual(detection.area, 1600)
        self.assertEqual(detection.solidity, 1.0)
        self.assertEqual(detection.shape, "Silhouette compacte")
        self.assertIsNone(detect_hand(mask, min_area=1601))

    def test_empty_mask_has_no_detection(self):
        self.assertIsNone(detect_hand(np.zeros((100, 100), dtype=np.uint8)))

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
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(result.mask.shape, frame.shape[:2])
        self.assertEqual(set(np.unique(result.mask)), {0, 255})
        display = draw_detection(frame, result.detection)
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
