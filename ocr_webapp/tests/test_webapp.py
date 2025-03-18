import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import pytest

from ocr_webapp import app
from config import Config
from custom_logger import CustomLogger
from custom_decorators import timeit

logger = CustomLogger(
    logger_log_level=Config.CLI_LOG_LEVEL,
    file_handler_log_level=Config.FILE_LOG_LEVEL,
    log_file_name=fr"{Config.ROOT_PATH}/ocr_webapp/tests/logs/test_logs.log"
).create_logger()


expected_results = {
    "koken.png": {
        "StatusCode": 200,
        "Prediction": "Dear",
    },

    "r06-111-00-00.png": {
        "StatusCode": 200,
        "Prediction": "Catherine",
    },

    "test.txt": {
        "StatusCode": 400,
        "Prediction": "File should be png or jpg",
    },
}


@pytest.fixture(scope="session", autouse=True)
def log_test_session():
    logger.info("Starting tests")
    yield
    logger.info("Finishing tests")

@pytest.fixture()
def app_client():
    app.config.update({
        "TESTING": True,
    })

    yield app

@timeit(logger=logger)
def test_index_route(app_client):
    response = app_client.test_client().get("/")

    assert response.status_code == 200
    logger.info("test_root_endpoint passed")

@timeit(logger=logger)
def test_upload(app_client):
    for image in list(expected_results.keys()):
        files = {"images": open(rf"{Config.TEST_IMAGES_FOLDER_PATH}/{image}", "rb")}
        response = app_client.test_client().post("/", data=files,
                                                    content_type="multipart/form-data",
                                                    follow_redirects=True)
        assert response.status_code == expected_results[image]["StatusCode"]
        assert expected_results[image]["Prediction"] in response.data.decode('utf-8')
    logger.info("test_upload passed")

