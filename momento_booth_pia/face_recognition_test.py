from pathlib import Path
import time

from PIL import Image
import face_recognition

if __name__ == "__main__":
    base_dir = Path("K:\\Pictures\\Photos\\2024-03-23 WuBDA Gala\\Difficult detection")
    export_dir = Path("K:\\Pictures\\Photos\\2024-03-23 WuBDA Gala\\Detection output")
    imgs = base_dir.glob("*.jpg")

    # yolo pose predict model=yolov8m-pose.pt source=''
    for img in imgs:
        # Load the jpg file into a numpy array
        image = face_recognition.load_image_file(img)

        # Find all the faces in the image using the default HOG-based model.
        # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
        # See also: find_faces_in_picture_cnn.py
        start = time.time()
        face_locations = face_recognition.face_locations(image)
        mid = time.time()
        print(f"Locating: {mid - start}")
        face_encodings = face_recognition.face_encodings(image, face_locations)
        end = time.time()
        print(f"Encodings: {end - mid}")

        print("I found {} face(s) in this photograph.".format(len(face_locations)))

        for face_location, face_encoding in zip(face_locations, face_encodings):

            # Print the location of each face in this image
            top, right, bottom, left = face_location
            print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

            # You can access the actual face itself like this:
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.show()
