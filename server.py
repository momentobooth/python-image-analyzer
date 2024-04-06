import argparse
import threading

from flask import Flask, request, send_from_directory, jsonify
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer
from PIL import Image
from momento_booth import *
from momento_booth_image_search import get_matching_images

app = Flask(__name__)
last_image = None
last_analysis = None
process_list = []
source_imgs = {}
data = {}


@app.route("/")
def hello_world():
    return ("<p>Post an image to /upload to analyze the faces in the photo and subsequently do a GET request to"
            "/get-matching-imgs to get the images that match with the faces in the previously uploaded image.</p>")


@app.route("/upload", methods=["POST"])
def upload_image():
    global last_image, last_analysis
    last_image = Image.open(request.files['file'].stream)
    last_analysis = get_faces([last_image]).encodings
    if len(last_analysis) == 0:
        return "No faces", 422
    return f"{len(last_analysis)}"


@app.route("/get-matching-imgs", methods=["GET"])
def get_matching_imgs():
    global last_image, last_analysis, data
    if last_image is None:
        return "No image was uploaded for analysis", 412
    if last_analysis is None:
        return "No faces were detected in last image", 412

    tolerance = request.args.get('tolerance', default=0.6, type=float)
    matching_images = get_matching_images(last_analysis, data, tolerance=tolerance)
    return jsonify(matching_images)


class CollageWatcherHandler(FileSystemEventHandler):
    def on_created(self, event: FileSystemEvent) -> None:
        print(f"Collage got created, {event.src_path}")
        process_list.append(Path(event.src_path))
        pass


class SourceWatcherHandler(FileSystemEventHandler):
    def on_created(self, event: FileSystemEvent) -> None:
        print(f"Source image got created, {event.src_path}")
        source_imgs[Path(event.src_path)] = False
        pass


def watcher(collage_path: Path, source_path: Path):
    observer = Observer()
    observer.schedule(CollageWatcherHandler(), path=str(collage_path), recursive=False)
    observer.schedule(SourceWatcherHandler(), path=str(source_path), recursive=False)
    observer.start()


def directory_type(raw_path: str) -> Path:
    p = Path(raw_path)
    if not p.is_dir():
        raise argparse.ArgumentTypeError('"{}" is not an existing directory'.format(raw_path))
    return p


def prepare_file_lists(collage_path: Path, source_path: Path):
    global process_list, source_imgs
    process_list = list(collage_path.glob("*.jpg"))
    source_imgs = {path: False for path in source_path.glob("*.jpg")}


def process_thread(data_file_path: Path):
    global process_list, source_imgs
    model = YOLO('yolov8x-pose-p6.pt')  # load the pretrained model

    while True:
        # If there are no images to process, sleep for a bit
        if len(process_list) == 0:
            time.sleep(0.1)
            continue

        collage_img_path = process_list.pop()
        already_processed = collage_img_path.name in data
        process_result = process_collage(collage_img_path, source_imgs, model, already_processed)
        if already_processed:
            print(f"Skipping {collage_img_path.name}, already in database")
            continue
        if process_result is not None:
            data[collage_img_path.name] = process_result
            save_data(data, data_file_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MomentoBooth image processing companion server")
    parser.add_argument("--collage-dir", type=directory_type, help="Output directory")
    parser.add_argument("--source-dir", type=directory_type, help="Input file(s) or directory")
    args = parser.parse_args()
    print(f"Running with \n - Collage dir: {args.collage_dir}\n - Source dir: {args.source_dir}")

    data_file_path = Path("data.json")
    data = load_data(data_file_path)
    watcher(args.collage_dir, args.source_dir)
    prepare_file_lists(args.collage_dir, args.source_dir)
    threading.Thread(target=process_thread, args=[data_file_path]).start()
    app.run(host="::", port=5000, debug=True, use_reloader=False)
