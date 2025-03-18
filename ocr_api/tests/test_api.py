import sys
import os
import io
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pytest
from fastapi.testclient import TestClient

from ocr_api import app
from config import Config
from custom_logger import CustomLogger

logger = CustomLogger(
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL,
    log_file_name=fr"{Config.ROOT_PATH}/ocr_api/tests/logs/test_logs.log"
).create_logger()

@pytest.fixture(scope="session", autouse=True)
def log_test_session():
    logger.info("Starting tests")
    yield
    logger.info("Finishing tests")
    
@pytest.fixture
def client():
    return TestClient(app)

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    logger.info("test_root_endpoint passed")

def test_predictions(client):
    test_files = [
        Path(f"{Config.TEST_IMAGES_FOLDER_PATH}/r06-111-00-00.png"),
        Path(f"{Config.TEST_IMAGES_FOLDER_PATH}/l03-004-00-02.png")
    ]
    
    files = []
    for file_path in test_files:
        with open(file_path, "rb") as f:
            file_content = f.read()
        files.append(
            ("files", (file_path.name, io.BytesIO(file_content), "image/png"))
        )
    response = client.post("/upload/", files=files)
    assert response.status_code == 200
    assert response.json()["res_list"] == ["Catherine", "took"]

    logger.info("test_predictions passed")
