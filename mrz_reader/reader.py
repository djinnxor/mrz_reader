import time

import cv2

from fastmrz import FastMRZ

from mrz_reader.segmentation import FaceDetection


class MRZReader:
    """Passport reader backed by fastmrz.

    fastmrz performs the whole MRZ path internally — MRZ-region segmentation
    (its own ONNX model), OCR-B text recognition (Tesseract with the `mrz`
    traineddata) and ICAO 9303 check-digit parsing. Face detection is done
    separately with the YuNet model.
    """

    def __init__(self,
                 tessdata_path: str = "./model/mrz_detector/tessdata",
                 facedetection_model: str = "./model/face_detector/face_detection_yunet_2023mar.onnx"):
        self.fast_mrz = FastMRZ(tessdata_path=tessdata_path)
        self.face_detection = FaceDetection(facedetection_model)

    def cleanup(self):
        try:
            if hasattr(self, 'face_detection') and self.face_detection:
                if hasattr(self.face_detection, 'detector'):
                    del self.face_detection.detector
                del self.face_detection
            if hasattr(self, 'fast_mrz') and self.fast_mrz:
                del self.fast_mrz
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def _run_mrz(self, img, timings):
        """Run fastmrz's segmentation + OCR, timing each model separately.

        fastmrz's public `get_details()` runs both stages in one call, so we
        drive its internals here to get a per-model breakdown. If those
        internals ever change shape, we fall back to the public call and only
        report a combined `mrz_segmentation_ocr` timing.
        """
        try:
            t0 = time.perf_counter()
            image_array = self.fast_mrz._process_image(img)
            self.fast_mrz.net.setInput(image_array)
            output_data = self.fast_mrz.net.forward()
            timings["mrz_segmentation"] = time.perf_counter() - t0

            # _get_roi crops the segmented region and runs Tesseract OCR-B on it.
            t0 = time.perf_counter()
            mrz_roi_text = self.fast_mrz._get_roi(output_data, img)
            timings["mrz_ocr"] = time.perf_counter() - t0

            return self.fast_mrz._cleanse_roi(mrz_roi_text) or ""
        except AttributeError as e:
            print(f"⚠️  fastmrz internals unavailable ({e}); timing segmentation+OCR together")
            t0 = time.perf_counter()
            raw_text = self.fast_mrz.get_details(img, input_type="numpy", ignore_parse=True) or ""
            timings["mrz_segmentation_ocr"] = time.perf_counter() - t0
            return raw_text

    def predict(self, image, do_facedetect=True):
        """Return (mrz_data, ocr_lines, face, timings).

        mrz_data  : fastmrz's parsed dict (fields + status + check digits)
        ocr_lines : the raw OCR-detected MRZ lines (top-to-bottom)
        face      : the detected face crop (BGR ndarray) or None
        timings   : per-stage inference time in seconds
        """
        timings = {}
        t_start = time.perf_counter()

        if isinstance(image, str):
            t0 = time.perf_counter()
            img = cv2.imread(image, cv2.IMREAD_COLOR)
            timings["image_read"] = time.perf_counter() - t0
        else:
            img = image

        if img is None:
            print("❌ Error: Input image is None. Check the image path or input data.")
            timings["total"] = time.perf_counter() - t_start
            return {"status": "FAILURE", "status_message": "Input image is None"}, [], None, timings

        # --- MRZ: fastmrz does segmentation + OCR internally (one OCR pass) ---
        try:
            raw_text = self._run_mrz(img, timings)
        except Exception as e:
            print(f"⚠️  fastmrz OCR failed: {e}")
            raw_text = ""

        ocr_lines = [line for line in raw_text.split("\n") if line.strip()]

        t0 = time.perf_counter()
        if raw_text.strip():
            try:
                # Parse the already-recognized text (input_type='text' → no second OCR).
                mrz_data = self.fast_mrz.get_details(raw_text, input_type="text")
            except Exception as e:
                print(f"⚠️  fastmrz parse failed: {e}")
                mrz_data = {
                    "status": "FAILURE",
                    "status_message": f"MRZ parse error: {e}",
                    "mrz_text": raw_text,
                }
        else:
            print("❌ No MRZ detected")
            mrz_data = {"status": "FAILURE", "status_message": "No MRZ detected"}
        timings["mrz_parse"] = time.perf_counter() - t0

        # --- Face detection (YuNet) on the same decoded image ---
        face = None
        if do_facedetect:
            t0 = time.perf_counter()
            try:
                face, _ = self.face_detection.detect(img)
            except Exception as e:
                print(f"⚠️  Face detection failed: {e}")
            timings["face_detection"] = time.perf_counter() - t0

        timings["total"] = time.perf_counter() - t_start
        return mrz_data, ocr_lines, face, timings
