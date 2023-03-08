# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 13:21:50 2022

@author: denis
"""

import mediapipe as mp


class Faces:
    mp_draw = mp.solutions.drawing_utils

    def __init__(self, model=0, min_detect_confidence=0.9,
                 verbose=False, boundary_size_px=50):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=model,
            min_detection_confidence=min_detect_confidence)
        self.verbose = verbose
        self.boundary_size_px = boundary_size_px

    def draw_face(self, img, results):
        if not results.detections:  # No Face was found
            if self.verbose:
                print('Face not found on input Image!')

            return False, img
        else:  # Draw face detections of each face.
            if self.verbose:
                print('Drawing on Faces!')

            annotated_image = img.copy()

            for detection in results.detections:
                self.mp_draw.draw_detection(annotated_image, detection)

            return True, annotated_image

    def detect_face(self, img):
        results = self.face_detection.process(img)

        if not results.detections:  # No Face was found
            if self.verbose:
                print('Face not found on input Image!')

            return False, results
        else:
            if self.verbose:
                print('One or more faces were found on input Image!')

            return True, results
