from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import uvicorn
import os
import cv2
import numpy as np
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import tempfile

load_dotenv()

app = FastAPI()

def load_models():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path="weights/RealESRGAN_x4plus.pth",
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=False
    )

    restorer = GFPGANer(
        model_path="weights/GFPGANv1.4.pth",
        upscale=4,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler
    )
    return restorer

restorer = load_models()

@app.get("/")
def health_check():
    return {"status": "ok"}

def inpaint_damage(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect tears — they show as very bright white/yellow streaks
    _, mask = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    inpainted = cv2.inpaint(img, mask, 7, cv2.INPAINT_TELEA)
    return inpainted

@app.post("/inpaint")
async def inpaint_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    result = inpaint_damage(img)
    
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, result)
        return FileResponse(tmp.name, media_type="image/png", filename="inpainted.png")

@app.post("/restore")
async def restore_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    #img = inpaint_tears(img)

    _, restored_faces, restored_img = restorer.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.5
    )

    if restored_img is None:
        restored_img = img

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, restored_img)
        return FileResponse(tmp.name, media_type="image/png", filename="restored.png")

#checky checky checky
