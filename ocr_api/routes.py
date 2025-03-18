from typing import List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import tensorflow as tf
from fastapi import UploadFile, HTTPException, File
from pydantic import BaseModel

from ocr_api import app, ocr_predictor, logger
from config import Config

class UploadResponse(BaseModel):
    res_list: List[str]


@app.get("/")
async def alive():
    return "Hello, I'm alive"


@app.post("/upload/", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)):
    logger.info(f"Uploaded {len(files)} files")
    if len(files) > Config.MAX_IMAGE_FILES:
        logger.error(f"(status code 400) Max number of files is {Config.MAX_IMAGE_FILES}")
        raise HTTPException(
            status_code=400,
            detail=f"Max number of files is {Config.MAX_IMAGE_FILES}"
        )
    try:
        logger.info("Preparing images")
        images = []

        for file in files:
            content = await file.read()
            image = tf.image.decode_png(content, 1)
            images.append(image)

        logger.info("Preparing predictions")
        predictions = ocr_predictor.get_predictions(
            images=images,
            img_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)
        )
        logger.info(f"{predictions=}")

        return UploadResponse(res_list=predictions)

    except HTTPException as http_ex:
        logger.error(f"HTTPException {http_ex}")
        raise http_ex
    except Exception as e:
        logger.error(f"(status code 500) Internal server error {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error {e}")