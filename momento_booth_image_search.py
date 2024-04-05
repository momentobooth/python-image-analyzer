import os
import time
from pathlib import Path

import face_recognition
import numpy as np
from PIL import Image
import piexif
import json

from momento_booth import *


if __name__ == "__main__":
    output_dir = Path("K:\\Pictures\\Photos\\2024-03-23 WuBDA Gala\\Output")
    test_img = Path("test-2.jpg")

    # Check if data.json exists
    data_file_path = Path("data.json")
    with open(data_file_path, "r") as f:
        data: dict = json.loads(f.read())

    test_encodings = get_faces([test_img]).encodings

    matches = []
    for (image, img_data) in data.items():
        np_encodings = [np.array(face) for face in img_data['faces']]
        image_match = False
        for test_encoding in test_encodings:
            results: list[bool] = face_recognition.compare_faces(np_encodings, test_encoding, tolerance=0.55)
            image_match = image_match or max(results)
            pass
        if image_match:
            matches.append(image)
            os.startfile(output_dir.joinpath(image))

    print("Completed analysis!")
