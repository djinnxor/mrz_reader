from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import base64
import cv2
import numpy as np
import time
import uvicorn
import os
import sys
import traceback
import gc  # Garbage collection
import psutil  # System monitoring

from mrz_reader.featurematching import FeatureMatching
from mrz_reader.perspective import Perspective
from mrz_reader.reader import MRZReader

app = FastAPI(
    title="MRZ Reader API",
    description="FastAPI service for passport MRZ recognition",
    version="1.0.0"
)

class PassportRequest(BaseModel):
    image_base64: str
    type: str
    do_facedetect: bool = True
    preprocess_config: Optional[Dict[str, Any]] = {
        "do_preprocess": True,
        "skewness": True,
        "delete_shadow": True,
        "clear_background": True
    }

class PassportResponse(BaseModel):
    success: bool
    mrz_data: Optional[Dict[str, Any]] = None
    face_base64: Optional[str] = None
    error: Optional[str] = None
    # Per-stage inference time in seconds (see `timings` keys in /predict).
    timings: Optional[Dict[str, float]] = None

def image_to_base64(image):
    if image is None:
        return None
    
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

def base64_to_image(base64_string):
    try:
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_bytes = base64.b64decode(base64_string)
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    except Exception as e:
        raise ValueError(f"Invalid base64 image format: {str(e)}")

@app.on_event("startup")
async def startup_event():
    global reader
    try:
        reader = MRZReader(
            tessdata_path="./model/mrz_detector/tessdata",
            facedetection_model="./model/face_detector/face_detection_yunet_2023mar.onnx",
        )
        print("✅ MRZ Reader initialized successfully")
    except Exception as e:
        print(f"⚠️  MRZ Reader initialization warning: {e}")
        print("📝 Note: Some model files may be missing. Face detection and segmentation may not work properly.")
        sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    global reader
    if reader:
        # Clean up resources
        try:
            if hasattr(reader, 'cleanup'):
                reader.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {e}")
    print("🛑 MRZ Reader shutdown completed")

@app.get("/")
async def root():
    return {"message": "MRZ Reader API is running", "status": "healthy"}

@app.post("/predict", response_model=PassportResponse)
async def predict_passport(request: PassportRequest):
    # Variables to track for cleanup
    image = None
    face = None
    request_start = time.perf_counter()

    try:
        if not reader:
            raise HTTPException(status_code=500, detail="MRZ Reader not initialized")

        try:
            t0 = time.perf_counter()
            image = base64_to_image(request.image_base64)
            image_decode_time = time.perf_counter() - t0
            if image is None:
                raise ValueError("Failed to decode image")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        # fastmrz handles MRZ segmentation, OCR-B recognition and check-digit
        # parsing internally; face detection (YuNet) runs on the same image.
        # `request.type` / `preprocess_config` are still accepted for backward
        # compatibility but are no longer used by the fastmrz path.
        print("🔍 Running MRZ recognition...")
        mrz_data, ocr_lines, face, timings = reader.predict(
            image,
            do_facedetect=request.do_facedetect,
        )
        print(f"✅ MRZ status: {mrz_data.get('status')}")

        t0 = time.perf_counter()
        face_base64 = None
        if face is not None and getattr(face, "size", 0) > 0:
            face_base64 = image_to_base64(face)
        face_encode_time = time.perf_counter() - t0

        timings = {
            "image_decode": image_decode_time,
            **timings,
            "face_encode": face_encode_time,
        }
        timings["inference_total"] = timings.pop("total", 0.0)
        timings["total"] = time.perf_counter() - request_start
        timings = {k: round(v, 4) for k, v in timings.items()}
        print("⏱️  " + ", ".join(f"{k}={v:.3f}s" for k, v in timings.items()))

        if image is not None:
            del image
        if face is not None:
            del face

        gc.collect()

        return PassportResponse(
            success=True,
            mrz_data=mrz_data,
            face_base64=face_base64,
            timings=timings,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in predict_passport: {e}")
        traceback.print_exc()
        return PassportResponse(
            success=False,
            error=str(e),
            timings={"total": round(time.perf_counter() - request_start, 4)},
        )
    finally:
        try:
            if 'image' in locals() and image is not None:
                del image
            if 'face' in locals() and face is not None:
                del face
        except:
            pass

@app.get("/health")
async def health_check():
    try:
        # Get memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        return {
            "status": "healthy",
            "reader_initialized": reader is not None,
            "version": "1.0.0",
            "memory_usage_mb": round(memory_info.rss / 1024 / 1024, 2),
            "cpu_percent": cpu_percent,
            "system_memory_percent": psutil.virtual_memory().percent
        }
    except ImportError:
        # Fallback if psutil is not available
        return {
            "status": "healthy",
            "reader_initialized": reader is not None,
            "version": "1.0.0"
        }

HTTP_PORT = int(os.getenv("PORT", 5006))
HTTP_HOST = os.getenv("HOST", "0.0.0.0")

def main():
    try:
        print("🚀 Starting MRZ Reader FastAPI server...")
        print(f"⚙️  Loading configuration: {HTTP_HOST}:{HTTP_PORT}")
        
        # Disable reload in production - major CPU issue fix!
        is_production = os.getenv("ENVIRONMENT", "production").lower() == "production"
        
        uvicorn.run(
            "main:app",
            host=HTTP_HOST,
            port=HTTP_PORT,
            reload=not is_production,  # Only reload in development
            log_level="warning" if is_production else "info",  # Reduce logging in production
            workers=1,  # Limit workers to prevent memory issues
            access_log=not is_production,  # Disable access logs in production
        )
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
