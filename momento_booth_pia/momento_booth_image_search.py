import os
from pathlib import Path

import numpy as np
import face_recognition
from momento_booth_pia.momento_booth import load_data, get_faces


def get_matching_images(test_encodings, data: dict[str, dict], tolerance=0.6) -> list[Path]:
    matches = []
    for (image, img_data) in data.items():
        encodings = img_data['faceEncodings']
        if len(encodings) == 0:
            continue
        image_match = False
        for test_encoding in test_encodings:
            results: list[bool] = face_recognition.compare_faces(encodings, test_encoding, tolerance=tolerance)
            image_match = image_match or max(results)
            pass
        if image_match:
            matches.append(image)
    return matches


if __name__ == "__main__":
    output_dir = Path("K:\\Pictures\\Photos\\2024-03-23 WuBDA Gala\\Output")
    test_img = Path("test-2.jpg")

    data_file_path = Path("data.json")
    data = load_data(data_file_path)

    test_encodings = get_faces([test_img]).encodings

    matches = get_matching_images(test_encodings, data)
    for image in matches:
        os.startfile(output_dir.joinpath(image))

    print("Completed analysis!")
