from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import uvicorn
import os
import cv2
import torch
import torch.nn.functional as F
from basicsr.archs.ddcolor_arch import DDColor
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import tempfile

load_dotenv()

app = FastAPI()

def load_sharpener():
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

    sharpener = GFPGANer(
        model_path="weights/GFPGANv1.4.pth",
        upscale=4,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=upsampler
    )
    return sharpener

def load_colorizer():
    model = DDColor(
        encoder_name='convnext-l',
        decoder_name='MultiScaleColorDecoder',
        input_size=[512,512],
        num_output_channels=2,
        last_norm='Spectral',
        do_normalize=False,
        num_queries=100,
        num_scales=3,
        dec_layers=9
    )
    ckpt = torch.load('weights/ddcolor.pth', map_location='cpu')
    model.load_state_dict(ckpt['params'], strict=False)
    model.eval()
    return model

def load_fixer():
    lama = torch.jit.load('weights/big-lama.pt', map_location='cpu')
    lama.eval()
    return lama

lama = load_fixer()
colorizer = load_colorizer()
sharpener = load_sharpener()

@app.post("/fix")
async def fix_image(file: UploadFile = File(...), mask: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    orig_size = (img.shape[1], img.shape[0])

    # Read mask
    mask_contents = await mask.read()
    mask_arr = np.frombuffer(mask_contents, np.uint8)
    mask_img = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)

    # Resize both to 512
    img_resized = cv2.resize(img, (512, 512))
    mask_resized = cv2.resize(mask_img, (512, 512), interpolation=cv2.INTER_NEAREST)

    # Binarize mask
    _, mask_binary = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

    # Convert image to RGB tensor
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    # Convert mask to tensor
    mask_tensor = torch.from_numpy(mask_binary).float() / 255.0
    mask_tensor = (mask_tensor > 0.5).float()
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)

    # Run LaMa
    with torch.no_grad():
        result = lama(img_tensor, mask_tensor)

    # Convert result back to image
    result = result.squeeze(0).permute(1, 2, 0).cpu().numpy()
    result = (result * 255).clip(0, 255).astype(np.uint8)
    result = cv2.resize(result, orig_size)
    result = result[:, :, ::-1].copy()

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, result)
        return FileResponse(tmp.name, media_type="image/png", filename="fixed.png")

@app.post("/colorize")
async def colorize_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Prepare tensor for DDColor
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize([512, 512])
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(img_pil).unsqueeze(0)

    # Run colorization
    with torch.no_grad():
        output = colorizer(tensor)

    # Extract UV channels
    uv = output.squeeze(0).permute(1, 2, 0).numpy()

    # Build LAB image and replace AB channels
    img_resized = cv2.resize(img, (512, 512))
    img_lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    ab = uv.copy()
    ab[:, :, 0] = (ab[:, :, 0] + 128).clip(0, 255)
    ab[:, :, 1] = (ab[:, :, 1] + 128).clip(0, 255)
    img_lab[:, :, 1] = ab[:, :, 0].astype(np.uint8)
    img_lab[:, :, 2] = ab[:, :, 1].astype(np.uint8)

    # Convert back to BGR and blend with original
    result = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    result = cv2.addWeighted(result, 0.6, img_resized, 0.4, 0)

    # Resize back to original dimensions
    result = cv2.resize(result, (img.shape[1], img.shape[0]))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, result)
        return FileResponse(tmp.name, media_type="image/png", filename="colorized.png")


@app.post("/sharpen")
async def sharpen_image(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run GFPGAN enhancement
    _, _, sharpened_img = sharpener.enhance(
        img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=0.8
    )

    if sharpened_img is None:
        sharpened_img = img

    # Apply unsharp mask for extra crispness
    gaussian = cv2.GaussianBlur(sharpened_img, (0, 0), 3)
    sharpened_img = cv2.addWeighted(sharpened_img, 1.8, gaussian, -0.8, 0)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        cv2.imwrite(tmp.name, sharpened_img)
        return FileResponse(tmp.name, media_type="image/png", filename="sharpened.png")
#checky checky checky
#uvicorn main:app --reload --port 3000
#source venv/Scripts/activate