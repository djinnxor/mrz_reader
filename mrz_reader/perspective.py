import cv2
import numpy as np

class Perspective:
    @staticmethod
    def transform(input_img, pts):
        if input_img is None:
            print("❌ Perspective: Input image is None")
            return None
            
        if pts is None or len(pts) != 4:
            print(f"❌ Perspective: Invalid points - need 4 corners, got {len(pts) if pts is not None else 0}")
            return None
        
        try:
            height, width = input_img.shape[:2]
            h = 800
            warp_h, warp_w = int(h * width / float(height)), int(h)
            
            if warp_h <= 0 or warp_w <= 0:
                print(f"❌ Perspective: Invalid output dimensions (h={warp_h}, w={warp_w})")
                return None
            
            pts_reshaped = pts.reshape(4, 2).astype(np.float32)
            
            res_boundary = np.float32([
                [0, 0],
                [0, warp_w],
                [warp_h, warp_w],
                [warp_h, 0]
            ])

            transform_matrix = cv2.getPerspectiveTransform(pts_reshaped, res_boundary)
            out_img = cv2.warpPerspective(input_img, transform_matrix, (warp_h, warp_w))
            
            print(f"✅ Perspective: Transformed from {width}x{height} to {warp_h}x{warp_w}")
            return out_img
            
        except Exception as e:
            print(f"❌ Perspective transform failed: {e}")
            return None
