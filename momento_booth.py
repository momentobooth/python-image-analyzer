import time
from pathlib import Path
from datetime import datetime, timedelta

import face_recognition
import numpy as np
from PIL import Image
import piexif
import json

from ultralytics import YOLO
import ultralytics.engine.results as yolo_results


def detect_people(model: YOLO, img: Path) -> yolo_results.Results:
    return model.predict(img, conf=0.5, verbose=False)[0]


class FacesResult:
    def __init__(self, encodings: np.ndarray, detection, encoding):
        self.encodings = encodings
        self.speed = {'detection': detection*1000, 'encoding': encoding*1000}


def detect_faces(image: Path | Image.Image):
    pillow_source_img = Image.open(image) if issubclass(type(image), Path) else image

    original_size = pillow_source_img.size
    pillow_source_img.thumbnail((600, 600))
    scaled_size = pillow_source_img.size
    np_image_small = np.array(pillow_source_img)

    scaling = original_size[0] / scaled_size[0]
    face_locations_raw = face_recognition.face_locations(np_image_small)
    face_locations = [tuple(int(v * scaling) for v in face) for face in face_locations_raw]
    return face_locations


def get_faces(sources: list[Path | Image.Image]) -> FacesResult:
    start_detecting_face = time.time()
    imgs_faces = [detect_faces(source) for source in sources]
    num_faces = [len(faces) for faces in imgs_faces]
    best_photo = num_faces.index(max(num_faces))

    # Load the jpg file into a numpy array
    pillow_source_img = Image.open(sources[best_photo]) if issubclass(type(sources[best_photo]), Path) else sources[best_photo]
    np_image = np.array(pillow_source_img)

    start_encoding_face = time.time()
    face_encodings = face_recognition.face_encodings(np_image, imgs_faces[best_photo])
    face_end = time.time()
    return FacesResult(face_encodings, start_encoding_face - start_detecting_face, face_end - start_encoding_face)


def find_image(name: str, source_imgs: dict):
    for source, taken in source_imgs.items():
        if not taken and name in source.name:
            source_imgs[source] = True
            return source
    return None


def date_from_exif(exif_dict: dict) -> datetime:
    date_raw = exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized].decode("utf-8")
    return datetime.strptime(date_raw, "%Y:%m:%d %H:%M:%S")


def process_collage(collage: Path, source_imgs: dict, model: YOLO, already_processed=False) -> (dict, None):
    start = time.time()
    pillow_collage_img = Image.open(collage)
    print(f"Opening image {collage.stem}")
    exif_dict = piexif.load(pillow_collage_img.info["exif"])
    json_str = exif_dict["Exif"][piexif.ExifIFD.MakerNote]
    json_obj = json.loads(json_str)
    print(f"\tJSON object: {json_obj}")
    raw_sources = json_obj["sourcePhotos"]
    found_sources = (find_image(source['filename'], source_imgs) for source in raw_sources)
    sources = [s for s in found_sources if s is not None]
    if already_processed:
        return None

    date_p = date_from_exif(exif_dict)
    date_s = date_p.strftime("%Y%m%d_%H")

    if len(sources) == 0:
        print("\tCould not find source files, skipping")
        return None
    else:
        print(f"\tImage has {len(sources)} source(s)")
    last_source = sources[-1]

    result = detect_people(model, last_source)
    print(f"\tDetected {len(result.boxes)} people in {sum(result.speed.values()):.1f} ms")

    face_results = get_faces(sources)
    print(f"\tFound {len(face_results.encodings)} face(s). Locating {face_results.speed['detection']:.1f} ms, encoding {face_results.speed['encoding']:.1f} ms.")
    end = time.time()

    return {
        "people": len(result.boxes),
        "faces": [encodings.tolist() for encodings in face_results.encodings]
    }


def load_data(file_path: Path) -> dict:
    if file_path.is_file():
        with open(file_path, "r") as f:
            return json.loads(f.read())
    else:
        return {}


def save_data(data:dict, file_path: Path):
    with open(file_path, "w") as f:
        f.write(json.dumps(data))


if __name__ == "__main__":
    output_dir = Path("K:\\Pictures\\Photos\\2024-03-23 WuBDA Gala\\Output")
    source_dir = Path("K:\\Pictures\\Photos\\2024-03-23 WuBDA Gala\\From camera")
    model = YOLO('yolov8x-pose-p6.pt')  # load the pretrained model

    # Check if data.json exists
    data_file_path = Path("data.json")
    data = load_data(data_file_path)

    output_imgs = output_dir.glob("*.jpg")
    source_imgs = {path: False for path in source_dir.glob("*.jpg")}
    i = 0
    for collage_img_path in output_imgs:
        already_processed = collage_img_path.name in data
        process_result = process_collage(collage_img_path, source_imgs, already_processed)
        if already_processed:
            print(f"Skipping {collage_img_path.name}, already in database")
            continue
        if process_result is not None:
            data[collage_img_path.name] = process_result
            i = i+1
        if i % 5 == 0:
            save_data(data, data_file_path)
            print(f"Saved {data_file_path.name}")
    save_data(data, data_file_path)
    print("Completed analysis!")
