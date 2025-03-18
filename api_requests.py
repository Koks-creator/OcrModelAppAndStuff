import requests
from config import Config

file_paths = [
    rf"{Config.TEST_IMAGES_FOLDER_PATH}\koken.png",
    rf"{Config.TEST_IMAGES_FOLDER_PATH}\koks.png"
]

files = []
for file_path in file_paths:
    with open(file_path, "rb") as file:
        files.append(('files', (file_path, file.read(), 'image/png')))

response = requests.post('http://localhost:5000/upload/', files=files)
print(response.json())
print(response.status_code)