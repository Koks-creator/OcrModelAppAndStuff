from pathlib import Path
import json
from typing import Union
import os
import logging


class Config:
    # Overall
    ROOT_PATH: str = Path(__file__).resolve().parent

    # Model
    MODELS_FOLDER_PATH: str = f"{ROOT_PATH}/models/"
    TEST_IMAGES_FOLDER_PATH: str = f"{ROOT_PATH}/images/"
    MODEL_NAME: str = "model1"
    IMAGE_WIDTH: int = 128
    IMAGE_HEIGHT: int = 32
    PADDING_TOKEN: int = 99

    # API
    OCR_PORT: int = 5000
    OCR_HOST: str = "127.0.0.1"
    MAX_IMAGE_FILES: int = 10
    OCR_API_LOG_FILE: str = f"{ROOT_PATH}/ocr_api/logs/api_logs.log"

    # WEB APP
    WEB_APP_PORT: int = 8000
    WEB_APP_HOST: str = "127.0.0.1"
    WEB_APP_DEBUG: bool = True
    WEB_APP_LOG_FILE: str = f"{ROOT_PATH}/ocr_webapp/logs/web_app.logs"
    WEB_APP_USE_SSL: bool = False
    WEB_APP_SSL_FOLDER: str = f"{ROOT_PATH}/ocr_webapp/ssl_cert"
    WEB_APP_TESTING: bool = False

    # LOGGER
    UVICORN_LOG_CONFIG_PATH: Union[str, os.PathLike, Path] = f"{ROOT_PATH}/ocr_api/uvicorn_log_config.json"
    CLI_LOG_LEVEL: int = logging.DEBUG
    FILE_LOG_LEVEL: int = logging.DEBUG

    def get_uvicorn_logger(self) -> dict:
        with open(self.UVICORN_LOG_CONFIG_PATH) as f:
            log_config = json.load(f)
            log_config["handlers"]["file_handler"]["filename"] = f"{Config.ROOT_PATH}/ocr_api/logs/api_logs.log"
            return log_config