import numpy as np
import cv2



class FaceDetection:
    def __init__(self, model_path, score_threshold=0.6, nms_threshold=0.3, top_k=5000):
        # YuNet (OpenCV Zoo) face detector. The input size is a placeholder here
        # and is set per-image in detect(), since passport crops vary in size.
        self.score_threshold = score_threshold
        self.detector = cv2.FaceDetectorYN_create(
            model_path, "", (320, 320), score_threshold, nms_threshold, top_k
        )

    def detect(self, image, confidence_input=None, expand_factor=0.25):
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_COLOR)
        else:
            img = image

        if img is None:
            print("❌ Face detection: received None image")
            return None, None

        # Honor a caller-supplied threshold; otherwise use the configured default.
        threshold = self.score_threshold if confidence_input is None else float(confidence_input)
        self.detector.setScoreThreshold(threshold)

        (h, w) = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)

        if faces is None or len(faces) == 0:
            print("⚠️  Face detection: no face found")
            return None, None

        # Pick the highest-scoring face (last column is the confidence score).
        best = faces[int(np.argmax(faces[:, -1]))]
        x, y, face_width, face_height = best[:4]
        confidence = float(best[-1])

        # Expand the box slightly for margin, then clamp to image bounds.
        expand_x = int(face_width * expand_factor)
        expand_y = int(face_height * expand_factor)

        new_startX = max(0, int(x) - expand_x)
        new_startY = max(0, int(y) - expand_y)
        new_endX = min(w, int(x + face_width) + expand_x)
        new_endY = min(h, int(y + face_height) + expand_y)

        if new_endX <= new_startX or new_endY <= new_startY:
            print(f"⚠️  Face detection: degenerate box after clamping (x={new_startX}:{new_endX}, y={new_startY}:{new_endY}), skipping")
            return None, None

        roi = img[new_startY:new_endY, new_startX:new_endX].copy()
        if roi.size == 0:
            print("⚠️  Face detection: empty ROI, skipping")
            return None, None

        return roi, confidence
