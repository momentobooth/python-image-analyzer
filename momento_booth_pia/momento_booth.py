import threading
import time
from pathlib import Path
from datetime import datetime, timedelta

import face_recognition
import numpy as np
from PIL import Image
import piexif
import json
from base64 import encodebytes, decodebytes
from filehash.filehash import FileHash, FileHashResult, VerifyHashResult

from ultralytics import YOLO
import ultralytics.engine.results as yolo_results

face_lock = threading.Lock()
np_dtype = np.float64


class FileChangedException(Exception):
    pass


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def detect_people(model: YOLO, img: Path) -> yolo_results.Results:
    return model.predict(img, conf=0.5, verbose=False)[0]


class FacesResult:
    def __init__(self, encodings: np.ndarray, locations: list[tuple[int, ...]], detection, encoding):
        self.encodings = encodings
        self.locations = locations
        self.speed = {'detection': detection*1000, 'encoding': encoding*1000}


def detect_faces(image: Path | Image.Image):
    pillow_source_img = Image.open(image) if issubclass(type(image), Path) else image.copy()

    original_size = pillow_source_img.size
    pillow_source_img.thumbnail((600, 600))
    scaled_size = pillow_source_img.size
    np_image_small = np.array(pillow_source_img)

    scaling = original_size[0] / scaled_size[0]
    face_locations_raw = face_recognition.face_locations(np_image_small)
    face_locations = [tuple(int(v * scaling) for v in face) for face in face_locations_raw]
    return face_locations


def get_faces(sources: list[Path | Image.Image], collage_path: Path | None = None) -> FacesResult:
    """
    Get faces and encodings from the best of a list of images.

    :param sources: Sources to look through
    :return: Results
    """
    face_lock.acquire(blocking=True)
    start_detecting_face = time.time()
    imgs_faces = [detect_faces(source) for source in sources]
    num_faces = [len(faces) for faces in imgs_faces]
    best_photo_index = num_faces.index(max(num_faces))

    best_photo = sources[best_photo_index]
    # Load the jpg file into a numpy array
    pillow_source_img = Image.open(best_photo) if issubclass(type(best_photo), Path) else best_photo
    np_image = np.array(pillow_source_img)

    if collage_path is not None:
        for (index, face) in enumerate(imgs_faces[best_photo_index]):
            top, right, bottom, left = face
            face_image = np_image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            filename = f"{collage_path.stem}-face-{index+1}.jpg"
            faces_dir = collage_path.parent.joinpath("faces")
            faces_dir.mkdir(exist_ok=True)
            pil_image.save(collage_path.parent.joinpath("faces", filename))

    start_encoding_face = time.time()
    face_encodings = np.array(face_recognition.face_encodings(np_image, imgs_faces[best_photo_index]), dtype=np_dtype)
    face_end = time.time()
    face_lock.release()
    return FacesResult(face_encodings, imgs_faces[best_photo_index], start_encoding_face - start_detecting_face, face_end - start_encoding_face)


def find_image(name: str, collage_path: Path, source_dir: Path) -> Path | None:
    source_imgs = list(source_dir.glob(f"*{name}"))
    if len(source_imgs) == 0:
        return None
    collage_time = date_from_image(collage_path)
    time_differences = [abs(collage_time - date_from_image(img) - timedelta(seconds=280)) for img in source_imgs]
    min_diff = min(time_differences)
    min_index = time_differences.index(min_diff)
    return source_imgs[min_index]


def date_from_image(image: Path):
    pillow_img = Image.open(image)
    return date_from_exif(piexif.load(pillow_img.info["exif"]))


def date_from_exif(exif_dict: dict) -> datetime:
    date_raw = exif_dict['Exif'][piexif.ExifIFD.DateTimeDigitized].decode("utf-8")
    return datetime.strptime(date_raw, "%Y:%m:%d %H:%M:%S")


def serialize_encodings(encodings: np.ndarray) -> str:
    """
    Serialize encodings in numpy format to a string.

    :param encodings: The array with encodings to be serialized
    :return: The serialized string
    """
    return encodebytes(encodings.astype(np_dtype).tobytes()).decode("ascii")


def deserialize_encodings(encodings: str) -> np.ndarray:
    """
    Parse encodings from a string to a numpy array.

    :param encodings: The string with encodings to be deserialized
    :return: The deserialized array
    """
    array = np.frombuffer(decodebytes(encodings.encode("ascii")), dtype=np_dtype)
    return array.reshape((array.size // 128, 128))


def save_img_with_maker_note(img_path: Path, maker_note: dict, img: Image.Image | None = None) -> None:
    """
    Save an image with given maker note in the exif metadata.

    :param img_path: Path to the collage image.
    :param maker_note: Dictionary with metadata about the image.
    :param img: Image to be saved. If not provided, the image will be loaded from the given path.
    """
    if img is None:
        img = Image.open(img_path)

    exif_dict = piexif.load(img.info["exif"])
    exif_dict["Exif"][piexif.ExifIFD.MakerNote] = json.dumps(maker_note).encode('utf-8')

    exif_bytes = piexif.dump(exif_dict)
    img.save(img_path, "jpeg", exif=exif_bytes)
    print(f"Saved maker note exif to {img_path.name}")


def process_collage(collage: Path, source_dir: Path, model: YOLO, already_processed=False) -> dict | None:
    hasher = FileHash("blake2b")
    start = time.time()
    print(f"Opening image {collage.stem}")
    hash_compare = hasher.hash_file(collage)

    pillow_collage_img = Image.open(collage)
    exif_dict = piexif.load(pillow_collage_img.info["exif"])
    json_str = exif_dict["Exif"][piexif.ExifIFD.MakerNote]
    json_obj: dict = json.loads(json_str)

    has_analysis = json_obj.get("faceEncodings", None) is not None
    if has_analysis:
        json_obj['faceEncodings'] = deserialize_encodings(json_obj['faceEncodings'])
        print("\tGot analysis from collage metadata")
        return json_obj
    if already_processed:
        if not has_analysis:
            print("\tAnalysis in data but not collage metadata, assuming new file")

    raw_sources = [source['filename'] for source in json_obj["sourcePhotos"]]
    found_sources = (find_image(source, collage, source_dir) for source in raw_sources)
    sources = [s for s in found_sources if s is not None]
    print(f"\toriginal sources: {raw_sources}, resolved = {[f.name for f in sources]}")

    if len(sources) == 0:
        print("\tCould not find source files, skipping")
        return None
    last_source = sources[-1]

    result = detect_people(model, last_source)
    print(f"\tDetected {len(result.boxes)} people in {sum(result.speed.values()):.1f} ms")
    if hasher.hash_file(collage) != hash_compare:
        raise FileChangedException("Hash mismatch")

    face_results = get_faces(sources, collage)
    print(f"\tFound {len(face_results.encodings)} face(s). Locating {face_results.speed['detection']:.1f} ms, encoding {face_results.speed['encoding']:.1f} ms.")
    end = time.time()

    # Add analysis results to the metadata
    json_obj.update({
        "peopleCount": len(result.boxes),
        "faceCount": len(face_results.encodings),
        "faceLocations": face_results.locations,
        "faceEncodings": serialize_encodings(face_results.encodings)
    })

    if hasher.hash_file(collage) != hash_compare:
        raise FileChangedException("Hash mismatch")
    # Save the metadata back to the file
    save_img_with_maker_note(collage, json_obj, pillow_collage_img)

    return_obj = dict(json_obj)
    return_obj['faceEncodings'] = face_results.encodings
    return json_obj


def load_data(file_path: Path) -> dict:
    if file_path.is_file():
        with open(file_path, "r") as f:
            try:
                return json.loads(f.read())
            except json.decoder.JSONDecodeError:
                print("Could not parse json from file, assuming empty")
                return {}
    else:
        return {}


def save_data(data:dict, file_path: Path):
    with open(file_path, "w") as f:
        f.write(json.dumps(data, cls=NumpyJsonEncoder))


if __name__ == "__main__":
    output_dir = Path("K:\\Pictures\\Photos\\2024-03-23 WuBDA Gala\\Output")
    source_dir = Path("K:\\Pictures\\Photos\\2024-03-23 WuBDA Gala\\From camera")
    model = YOLO('yolov8x-pose-p6.pt')  # load the pretrained model

    # Check if data.json exists
    data_file_path = Path("data.json")
    data = load_data(data_file_path)

    output_imgs = output_dir.glob("*.jpg")
    source_imgs = list(source_dir.glob("*.jpg"))
    i = 0
    for collage_img_path in output_imgs:
        already_processed = collage_img_path.name in data

        while True:
            try:
                process_result = process_collage(collage_img_path, source_dir, model, already_processed)
                break
            except FileChangedException as e:
                print(f"{collage_img_path.name} changed during processing, retrying...")
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
