from flask import Flask, request, send_from_directory, jsonify
from PIL import Image
from momento_booth import *
from momento_booth_image_search import get_matching_images

app = Flask(__name__)
last_image = None
last_analysis = None
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


if __name__ == '__main__':
    data_file_path = Path("data.json")
    with open(data_file_path, "r") as f:
        data: dict = json.loads(f.read())
    app.run(host="::", port=5000, debug=True)
