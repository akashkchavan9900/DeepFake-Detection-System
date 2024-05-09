import os
import re
import sys
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from detection import test_full_image_network

app = FastAPI()

TEMP_DIR = "./temp"

# Create temp directory if it doesn't exist
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/detect/")
async def detect_fake_video(video_file: UploadFile = File(...)):
    if not video_file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Invalid file format. Must be an MP4 video.")

    # Save the uploaded video file temporarily
    temp_file_path = f"./temp/{video_file.filename}"
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(video_file.file, temp_file)

    model_path = './models/x-model23.p'
    output_path = './results/'
    fast = True

    # Perform fake video detection
    prediction = test_full_image_network(temp_file_path, model_path, output_path, fast)

    # Clean up temporary file
    os.remove(temp_file_path)

    return {"prediction": prediction}
