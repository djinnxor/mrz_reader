import cv2
import numpy as np
from .template import Templates

MIN_SHAPE = (400, 600)
RATIO = 0.75
MIN_GOOD_MATCHES = 50

class FeatureMatching:
    def __init__(self, temp_path):
        self.templates = Templates(temp_path).load()
        self.feature_detector = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.03, edgeThreshold=15)
        self.kp_objects = []
        self.dest_objects = []
        self.matches_mask = None
        for temp in self.templates:
            kp_object, dest_object = self.feature_detector.detectAndCompute(
                temp.img, None)
            self.kp_objects.append(kp_object)
            self.dest_objects.append(dest_object)

    def __findBoundaryPoints(self, good_mtch, kp_scene, index):
        src_pts = np.float32(
            [self.kp_objects[index][m.queryIdx].pt for m in good_mtch]).reshape(-1, 1, 2)
        dest_pts = np.float32(
            [kp_scene[m.trainIdx].pt for m in good_mtch]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dest_pts, cv2.RANSAC, 5.0)
        self.matches_mask = mask.ravel().tolist()

        h_1, w_1 = self.templates[index].img.shape[0] - \
            1, self.templates[index].img.shape[1] - 1
        bound_pts = np.float32(
            [[0, 0], [0, h_1], [w_1, h_1], [w_1, 0]]).reshape(-1, 1, 2)
        dest_bound_pts = cv2.perspectiveTransform(bound_pts, M)
        return dest_bound_pts

    def match_template(self, input_image, index):
        enhanced = self._enhance_for_matching(input_image)
        scaled_imgs = list(FeatureMatching.make_scaled_ims(enhanced))
        
        print(f"🔍 Feature Matching: Testing {len(scaled_imgs)} scaled images")
        
        best_matches = 0
        best_result = None
        
        for idx, image in enumerate(scaled_imgs):
            kp_scene, dest_scene = self.feature_detector.detectAndCompute(image, None)

            if dest_scene is None or len(dest_scene) < 2:
                print(f"⚠️  Scale {idx}: No features detected")
                continue
                
            matches = FeatureMatching.__FlannMethod(self.dest_objects[index], dest_scene)
            good_mtch = FeatureMatching.__FindGoodMatches(RATIO, matches)
            
            print(f"📊 Scale {idx}: Found {len(good_mtch)} good matches (need {MIN_GOOD_MATCHES})")

            if len(good_mtch) > best_matches:
                best_matches = len(good_mtch)
                if len(good_mtch) >= MIN_GOOD_MATCHES:
                    best_result = self.__findBoundaryPoints(good_mtch, kp_scene, index)

            if len(good_mtch) >= MIN_GOOD_MATCHES:
                dest_border_points = self.__findBoundaryPoints(good_mtch, kp_scene, index)
                print(f"✅ Feature Matching: Success with {len(good_mtch)} matches")
                return dest_border_points

        if best_result is not None and best_matches >= 30:
            print(f"⚠️  Using best match with {best_matches} matches (below threshold but acceptable)")
            return best_result
            
        print(f"❌ Feature Matching: Failed - best match had only {best_matches} matches")
        return None
    
    def _enhance_for_matching(self, image):
        if image is None or len(image.shape) < 2:
            return image
            
        try:
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                enhanced = image.copy()
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(enhanced)
            
            kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
            
            return enhanced
        except Exception as e:
            print(f"⚠️  Enhancement failed: {e}, using original")
            return image

    @staticmethod
    def make_scaled_ims(img, min_shape=MIN_SHAPE):
        ratio = RATIO
        shape = (img.shape[0] / ratio, img.shape[1] / ratio)
        
        yield img
        
        scale_count = 0
        while scale_count < 8:
            shape = (int(shape[0] * ratio), int(shape[1] * ratio))
            if shape[0] < min_shape[0] or shape[1] < min_shape[1]:
                break
            scaled = cv2.resize(img, (shape[1], shape[0]))
            yield scaled
            scale_count += 1

    @staticmethod
    def __FlannMethod(dest_obj, dest_scene):
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        
        if dest_obj is None or dest_scene is None:
            return []
        if len(dest_obj) < 2 or len(dest_scene) < 2:
            return []
            
        try:
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(dest_obj.astype(
                np.float32), dest_scene.astype(np.float32), k=2)
            return matches
        except Exception as e:
            return []

    @staticmethod
    def __FindGoodMatches(ratio, matches):
        good_matches = []
        if not matches:
            return good_matches
            
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
        return good_matches
