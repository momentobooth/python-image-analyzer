import argparse
import threading
from pathlib import Path
import time

from flask import Flask, request, send_from_directory, jsonify
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer
from PIL import Image
from ultralytics import YOLO
from momento_booth_pia.momento_booth import process_collage, get_faces, save_data, load_data, FacesResult, FileChangedException
from momento_booth_pia.momento_booth_image_search import get_matching_images

app = Flask(__name__)
last_image = None
last_analysis: FacesResult | None = None
process_list = []
source_imgs = []
data = {}


@app.route("/")
def hello_world():
    return ("<p>Post an image to /upload to analyze the faces in the photo and subsequently do a GET request to"
            "/get-matching-imgs to get the images that match with the faces in the previously uploaded image.</p>")


@app.route("/upload", methods=["POST"])
def upload_image():
    global last_image, last_analysis
    last_image = Image.open(request.files['file'].stream)
    save_location = args.source_dir.parent.joinpath("upload.jpg")
    last_image.save(save_location)
    last_analysis = get_faces([last_image], save_location)
    if len(last_analysis.encodings) == 0:
        return "No faces", 422
    print(f"Found {len(last_analysis.encodings)} faces. {last_analysis.encodings[0][0:4]}")
    return jsonify({
        "num": len(last_analysis.encodings),
        "locations": last_analysis.locations,
    })


@app.route("/get-matching-imgs", methods=["GET"])
def get_matching_imgs():
    global last_image, last_analysis, data
    if last_image is None:
        return "No image was uploaded for analysis", 412
    if last_analysis is None:
        return "No faces were detected in last image", 412

    tolerance = request.args.get('tolerance', default=0.6, type=float)
    matching_images = get_matching_images(last_analysis.encodings, data, tolerance=tolerance)
    print(f"Found {len(matching_images)} matches.")
    return jsonify(matching_images)


class CollageWatcherHandler(FileSystemEventHandler):
    def on_created(self, event: FileSystemEvent, log=True):
        path = Path(event.src_path)
        if path.is_dir() or (path.suffix.lower() not in ['.jpg', '.jpeg', '.png']):
            return
        if log:
            print(f"Collage got created, {event.src_path}")
        process_list.append(path)
        pass

    def on_modified(self, event: FileSystemEvent):
        path = Path(event.src_path)
        if path.is_dir() or (path.suffix.lower() not in ['.jpg', '.jpeg', '.png']):
            return
        print(f"Collage was modified, {event.src_path}")
        # self.on_deleted(event, False)
        # self.on_created(event, False)

    def on_deleted(self, event: FileSystemEvent, log=True):
        path = Path(event.src_path)
        if path.is_dir() or (path.suffix.lower() not in ['.jpg', '.jpeg', '.png']):
            return
        del data[path.name]
        save_data(data, data_file_path)
        if log:
            print(f"Collage got removed, {event.src_path}")


class SourceWatcherHandler(FileSystemEventHandler):
    def on_created(self, event: FileSystemEvent) -> None:
        print(f"Source image got created, {event.src_path}")
        source_imgs.append(Path(event.src_path))
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
    source_imgs = list(source_path.glob("*.jpg"))


def process_thread(data_file_path: Path, collage_path: Path, source_path: Path, model_path: Path | str):
    global process_list, source_imgs
    model = YOLO(model_path)  # load the pretrained model

    while True:
        # If there are no images to process, sleep for a bit
        if len(process_list) == 0:
            time.sleep(0.1)
            continue

        collage_img_path = process_list.pop(0)
        already_processed = collage_img_path.name in data
        while True:
            try:
                process_result = process_collage(collage_img_path, source_path, model, already_processed)
                break
            except FileChangedException as e:
                print(f"{collage_img_path.name} changed during processing, retrying...")
        if already_processed:
            print(f"Skipping {collage_img_path.name}, already in database")
            continue
        if process_result is not None:
            data[collage_img_path.name] = process_result
            save_data(data, data_file_path)


def main():
    global args, data, data_file_path
    parser = argparse.ArgumentParser(description="MomentoBooth image processing companion server")
    parser.add_argument("-c", "--collage-dir", type=directory_type, help="Collage directory")
    parser.add_argument("-s", "--source-dir", type=directory_type, help="Input file(s) or directory")
    parser.add_argument("-p", "--port", default=3232, type=int, help="Port to run the server on")
    parser.add_argument("--host", default="localhost",
                        help="""Host to run the server on. Use "::" to accept requests for all interfaces""")
    parser.add_argument("-m", "--model", default="yolov8x-pose-p6.pt", help="Choose the model used for YOLOv8")
    args = parser.parse_args()
    print(f"Running with \n - Collage dir: {args.collage_dir}\n - Source dir: {args.source_dir}")

    # Could also use Path(__file__).parent as folder
    data_file_path = args.collage_dir.joinpath("data.json")
    data = load_data(data_file_path)
    watcher(args.collage_dir, args.source_dir)
    prepare_file_lists(args.collage_dir, args.source_dir)
    model_path = Path(__file__).parent.parent.joinpath("models", args.model)
    threading.Thread(target=process_thread,
                     args=[data_file_path, args.collage_dir, args.source_dir, model_path]
                     ).start()
    app.run(host=args.host, port=args.port, debug=True, use_reloader=False)


if __name__ == '__main__':
    main()
