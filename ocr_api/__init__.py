import sys
import os
from pathlib import Path
from logging import Logger
sys.path.append(str(Path(__file__).resolve().parent.parent))
from fastapi import FastAPI
from ocr_tools.ocr_predictor import OcrPredictor
from config import Config
from custom_logger import CustomLogger


def setup_logging() -> Logger:
    """Configure logging for the api"""
    log_dir = os.path.dirname(Config.WEB_APP_LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = CustomLogger(
        logger_name="middleware_logger",
        logger_log_level=Config.CLI_LOG_LEVEL,
        file_handler_log_level=Config.FILE_LOG_LEVEL,
        log_file_name=Config.OCR_API_LOG_FILE
    ).create_logger()

    return logger

logger = setup_logging()

logger.info("Starting API")
app = FastAPI(title="OcrApi")
logger.info("Starting predictor")
ocr_predictor = OcrPredictor(
    model_folder_path=f"{Config.MODELS_FOLDER_PATH}{Config.MODEL_NAME}"
)
logger.info("Started")

from ocr_api import routes