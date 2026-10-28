"""DRAFT: MRZReader + one added stage — feature-match/align the decoded image.

Everything here is `MRZReader` verbatim: same models, same config, same
fastmrz segmentation net, same Tesseract OCR-B pass, same parser, same YuNet
face detection, same order, same timing keys. `FeatureMatchMRZReader` subclasses
`MRZReader`, so `__init__` config and `_run_mrz` are literally inherited rather
than copied.

The one and only difference is inserted right after the input image is decoded:

    decoded image -> feature-match against template passports (ORB/SIFT)
                  -> RANSAC homography input -> template
                  -> warp the WHOLE page into the template's geometry
                  -> ...from here on, the unmodified reader pipeline

So the alignment normalizes rotation, perspective and scale before the
segmentation net and OCR ever see the image. If no template matches, or the
homography looks degenerate, the original image is passed through untouched and
the pipeline degrades to exactly `MRZReader`.

Run the comparison:

    .venv/bin/python draft_feature_match.py --runs 5
    .venv/bin/python draft_feature_match.py --detector SIFT --save-debug out/

CAVEAT: the only passport images in this repo are the three under test/, and
they double as the templates. A self-match is meaningless, so any template whose
path equals the input's is skipped (leave-one-out) — which leaves two templates
per input. That is far too small a corpus to conclude anything about accuracy.
"""

import argparse
import os
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from mrz_reader.reader import MRZReader

DEFAULT_TEMPLATES = [
    "test/NEW_PASSPORT.jpg",
    "test/OLD_PASSPORT.jpg",
]
DEFAULT_INPUTS = [
    "test/passport_test.jpg",
    "test/NEW_PASSPORT.jpg",
    "test/OLD_PASSPORT.jpg",
]

# Same defaults as MRZReader.__init__ — repeated only so the CLI can pass them.
TESSDATA_PATH = "./model/mrz_detector/tessdata"
FACEDETECTION_MODEL = "./model/face_detector/face_detection_yunet_2023mar.onnx"

# Lowe's ratio threshold, and the minimum match/inlier counts before we trust a
# homography (4 is the mathematical minimum; these keep RANSAC from fitting noise).
LOWE_RATIO = 0.75
MIN_GOOD_MATCHES = 12
MIN_INLIERS = 15


@dataclass
class Template:
    """A reference passport page with its features precomputed."""

    path: str
    size: Tuple[int, int]  # (width, height)
    keypoints: tuple
    descriptors: np.ndarray


def build_detector(name: str):
    """Return (detector, matcher) for the requested feature type."""
    if name.upper() == "SIFT":
        # SIFT descriptors are float -> L2.
        return cv2.SIFT_create(nfeatures=2000), cv2.BFMatcher(cv2.NORM_L2)
    # ORB descriptors are binary -> Hamming.
    return cv2.ORB_create(nfeatures=2000), cv2.BFMatcher(cv2.NORM_HAMMING)


class FeatureMatchMRZReader(MRZReader):
    """MRZReader with a feature-matching alignment stage on the decoded image."""

    def __init__(
        self,
        template_paths: List[str],
        detector_name: str = "ORB",
        tessdata_path: str = TESSDATA_PATH,
        facedetection_model: str = FACEDETECTION_MODEL,
    ):
        # Inherit the reader's exact model/config setup.
        super().__init__(tessdata_path=tessdata_path, facedetection_model=facedetection_model)

        self.detector_name = detector_name.upper()
        self.detector, self.matcher = build_detector(self.detector_name)
        self.templates: List[Template] = []

        # Template features are computed once at startup, so a request only pays
        # for input features + N template matches + one warp.
        for path in template_paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"⚠️  Template unreadable, skipping: {path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            if descriptors is None or len(keypoints) < MIN_GOOD_MATCHES:
                print(f"⚠️  Too few {self.detector_name} features, skipping template: {path}")
                continue

            h, w = img.shape[:2]
            self.templates.append(Template(path, (w, h), keypoints, descriptors))

        if not self.templates:
            raise RuntimeError("No usable templates — cannot run the matcher.")
        print(f"🧩 {len(self.templates)} template(s) loaded ({self.detector_name})")

    def _match_template(self, template: Template, descriptors: np.ndarray) -> List:
        """Lowe-ratio-filtered matches between one template and the input."""
        if descriptors is None or len(descriptors) < 2:
            return []
        pairs = self.matcher.knnMatch(descriptors, template.descriptors, k=2)
        # knnMatch can return 1-element lists when a descriptor has no second
        # neighbour; those can't be ratio-tested, so drop them.
        return [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < LOWE_RATIO * pair[1].distance
        ]

    @staticmethod
    def _is_sane(homography: np.ndarray, img_shape, template_size) -> bool:
        """Reject homographies that would smear the page instead of aligning it.

        RANSAC will happily return a fit from 15 coincidental matches; warping
        through it produces a streak, and every downstream stage then fails on
        garbage. Cheaper to detect here: map the input's corners and require a
        convex quad covering a plausible share of the template canvas.
        """
        h, w = img_shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(corners, homography)
        if not np.isfinite(mapped).all():
            return False

        pts = mapped.reshape(-1, 2)
        if not cv2.isContourConvex(pts.astype(np.float32)):
            return False

        tw, th = template_size
        coverage = abs(cv2.contourArea(pts.astype(np.float32))) / float(tw * th)
        return 0.4 <= coverage <= 4.0

    def align(self, img: np.ndarray, exclude_path: Optional[str] = None):
        """Warp the decoded image into the best-matching template's geometry.

        Returns (aligned_image, info). On any failure the ORIGINAL image is
        returned, so the caller's pipeline is unchanged from MRZReader's.
        """
        info = {"template": None, "inliers": 0, "good_matches": 0, "aligned": False}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        info["keypoints"] = len(keypoints) if keypoints else 0
        if descriptors is None:
            return img, info

        best = None  # (inliers, template, homography)
        for template in self.templates:
            if exclude_path and os.path.abspath(template.path) == os.path.abspath(exclude_path):
                continue  # never self-match

            good = self._match_template(template, descriptors)
            if len(good) < MIN_GOOD_MATCHES:
                continue

            # queryIdx indexes the input, trainIdx the template: input -> template.
            src = np.float32([keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([template.keypoints[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if homography is None:
                continue

            inliers = int(mask.sum())
            if best is None or inliers > best[0]:
                best = (inliers, template, homography)
                info["good_matches"] = len(good)

        if best is None or best[0] < MIN_INLIERS:
            return img, info

        inliers, template, homography = best
        info.update({"template": template.path, "inliers": inliers})

        if not self._is_sane(homography, img.shape, template.size):
            info["rejected"] = "implausible homography"
            return img, info

        aligned = cv2.warpPerspective(img, homography, template.size)
        info["aligned"] = True
        return aligned, info

    def predict(self, image, do_facedetect=True, exclude_path=None):
        """MRZReader.predict with the alignment stage inserted after decode.

        Identical to the parent apart from the `feature_match` block and the
        extra timing key it adds; returns the same 4-tuple.
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

        # --- THE ONLY ADDED STAGE: align the decoded image to a template ---
        t0 = time.perf_counter()
        img, match_info = self.align(img, exclude_path=exclude_path)
        timings["feature_match"] = time.perf_counter() - t0
        if not match_info["aligned"]:
            print(f"⚠️  No usable template match ({match_info.get('rejected', 'below threshold')}); "
                  "continuing on the original image")

        # --- everything below is MRZReader.predict, unchanged ---

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
        timings["_match_info"] = match_info  # non-timing payload, stripped before printing
        return mrz_data, ocr_lines, face, timings


# --------------------------------------------------------------------------
# Comparison harness
# --------------------------------------------------------------------------

# Fields worth diffing between the two pipelines; the rest of fastmrz's dict is
# status/check-digit bookkeeping.
COMPARE_FIELDS = [
    "document_type",
    "country_code",
    "surname",
    "given_name",
    "document_number",
    "nationality",
    "date_of_birth",
    "sex",
    "expiry_date",
]

# Stages present in both pipelines, in reader order.
SHARED_STAGES = [
    "mrz_segmentation",
    "mrz_segmentation_ocr",
    "mrz_ocr",
    "mrz_parse",
    "face_detection",
    "total",
]


def rotate_expand(img: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate about the centre, growing the canvas so no corner is clipped.

    Used by --rotate to synthesize a misaligned input. Clipping would destroy
    MRZ pixels outright, and no alignment stage can recover deleted data — the
    canvas has to grow for the test to measure what it claims to.
    """
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += (new_w / 2) - w / 2
    matrix[1, 2] += (new_h / 2) - h / 2
    return cv2.warpAffine(img, matrix, (new_w, new_h), borderValue=(255, 255, 255))


def median_timings(runs: List[dict]) -> Dict[str, float]:
    """Median of each timing key across repeated runs."""
    keys = [k for k in runs[0] if not k.startswith("_")]
    return {k: statistics.median(r[k] for r in runs if k in r) for k in keys}


def compare_fields(a: dict, b: dict) -> List[str]:
    """Field names where the two parsed results disagree."""
    return [f for f in COMPARE_FIELDS if (a or {}).get(f, "") != (b or {}).get(f, "")]


def run_comparison(args) -> None:
    baseline = MRZReader(tessdata_path=TESSDATA_PATH, facedetection_model=FACEDETECTION_MODEL)
    matched = FeatureMatchMRZReader(args.templates, detector_name=args.detector)

    if args.save_debug:
        os.makedirs(args.save_debug, exist_ok=True)

    summary = []

    for path in args.inputs:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️  Input unreadable, skipping: {path}")
            continue

        if args.rotate:
            img = rotate_expand(img, args.rotate)

        print(f"\n{'=' * 72}\n📄 {path}  {img.shape[1]}x{img.shape[0]}"
              f"{f'  (rotated {args.rotate}°)' if args.rotate else ''}\n{'=' * 72}")
        if not args.self_match and any(
            os.path.abspath(t) == os.path.abspath(path) for t in args.templates
        ):
            print("⚠️  This input is also a template; it is excluded from its own match "
                  "(leave-one-out), so one fewer template is available here.")

        # With --rotate the input has been transformed, so matching it against
        # its own source image is a legitimate test rather than a self-match.
        exclude = None if args.self_match else path

        base_runs, fm_runs = [], []
        base_data = fm_data = None
        base_lines = fm_lines = []
        base_face = fm_face = None
        info = None

        for _ in range(args.runs):
            base_data, base_lines, base_face, t = baseline.predict(
                img, do_facedetect=args.facedetect
            )
            base_runs.append(t)

            fm_data, fm_lines, fm_face, t = matched.predict(
                img, do_facedetect=args.facedetect, exclude_path=exclude
            )
            info = t.pop("_match_info", info)
            fm_runs.append(t)

        b_med = median_timings(base_runs)
        f_med = median_timings(fm_runs)

        if info:
            state = "aligned" if info["aligned"] else f"NOT aligned ({info.get('rejected', 'below threshold')})"
            print(f"🧩 {state} | template={info['template']} "
                  f"inliers={info['inliers']}/{info['good_matches']} good "
                  f"keypoints={info.get('keypoints')}")

        print(f"\n-- timings (median of {args.runs} run(s), seconds) --")
        print(f"{'stage':<24}{'reader':>10}{'+match':>10}{'delta':>10}")
        print(f"{'feature_match':<24}{'-':>10}{f_med.get('feature_match', 0):>10.4f}"
              f"{f_med.get('feature_match', 0):>+10.4f}")
        for stage in SHARED_STAGES:
            b_v, f_v = b_med.get(stage), f_med.get(stage)
            if b_v is None or f_v is None:
                continue
            print(f"{stage:<24}{b_v:>10.4f}{f_v:>10.4f}{f_v - b_v:>+10.4f}")

        print("\n-- results --")
        print(f"reader status: {base_data.get('status')}   lines: {len(base_lines)}   "
              f"face: {'yes' if base_face is not None else 'no'}")
        print(f"+match status: {fm_data.get('status')}   lines: {len(fm_lines)}   "
              f"face: {'yes' if fm_face is not None else 'no'}")
        for label, lines in [("reader", base_lines), ("+match", fm_lines)]:
            for line in lines:
                print(f"  {label} | {line}")

        differing = compare_fields(base_data, fm_data)
        if base_data.get("status") == "SUCCESS" and fm_data.get("status") == "SUCCESS":
            if differing:
                print(f"❌ parsed fields differ: {', '.join(differing)}")
                for f in differing:
                    print(f"   {f}: reader={base_data.get(f)!r} +match={fm_data.get(f)!r}")
            else:
                print("✅ parsed fields identical")
        else:
            print("⚠️  field comparison skipped (at least one pipeline did not succeed)")

        if args.save_debug:
            aligned, _ = matched.align(img, exclude_path=exclude)
            out = os.path.join(
                args.save_debug, f"{os.path.splitext(os.path.basename(path))[0]}_aligned.png"
            )
            cv2.imwrite(out, aligned)
            print(f"💾 aligned image -> {out}")

        summary.append({
            "image": path,
            "aligned": bool(info and info["aligned"]),
            "base_total": b_med.get("total"),
            "fm_total": f_med.get("total"),
            "base_status": base_data.get("status"),
            "fm_status": fm_data.get("status"),
            "fields_differ": len(differing),
        })

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    print(f"{'image':<26}{'aligned':>9}{'total r/+m':>18}{'status r/+m':>24}")
    for s in summary:
        total = f"{s['base_total']:.3f}/{s['fm_total']:.3f}"
        status = f"{s['base_status']}/{s['fm_status']}"
        print(f"{os.path.basename(s['image']):<26}{str(s['aligned']):>9}{total:>18}{status:>24}")

    baseline.cleanup()
    matched.cleanup()


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES)
    p.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    p.add_argument("--detector", default="ORB", choices=["ORB", "SIFT", "orb", "sift"])
    p.add_argument("--runs", type=int, default=3, help="repeat each pipeline N times, report medians")
    p.add_argument("--facedetect", action="store_true", help="include the YuNet stage")
    p.add_argument("--rotate", type=float, default=0.0, metavar="DEG",
                   help="rotate each input by DEG (expanding the canvas) before both "
                        "pipelines — synthesizes a misaligned document")
    p.add_argument("--self-match", action="store_true",
                   help="allow an input to match its own source template; only "
                        "meaningful together with --rotate")
    p.add_argument("--save-debug", metavar="DIR", help="write aligned images to DIR")
    return p.parse_args()


if __name__ == "__main__":
    run_comparison(parse_args())
