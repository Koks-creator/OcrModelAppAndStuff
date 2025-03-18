from typing import List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import tensorflow as tf
from fastapi import FastAPI, UploadFile, HTTPException, Request, File
from pydantic import BaseModel
import uvicorn

from ocr_tools.ocr_predictor import OcrPredictor
from config import Config


app = FastAPI(title="OcrApi")
ocr_predictor = OcrPredictor(
    model_folder_path=f"{Config.MODELS_FOLDER_PATH}{Config.MODEL_NAME}"
)

class UploadResponse(BaseModel):
    res_list: List[str]

MAX_FILES = 5
@app.post("/upload/", response_model=UploadResponse)
async def upload(files: List[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Max number of files is {MAX_FILES}."
        )
    try:
        predictions_list = []
        images = []

        # Przetwarzamy każdy nadesłany plik
        for file in files:
            content = await file.read()
            image = tf.image.decode_png(content, 1)
            images.append(image)

        predictions = ocr_predictor.get_predictions(
            images=images,
            img_size=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)
        )

        return UploadResponse(res_list=predictions)

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal server error")
    

if __name__ == '__main__':
    # log_config=uvicorn_log_config
    #  log_config=Config().get_uvicorn_logger()
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)