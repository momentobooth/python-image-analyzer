from pathlib import Path
import time

from ultralytics import YOLO
import ultralytics.engine.results as yolo_results
import cv2

if __name__ == "__main__":
    # Load a model
    # model = YOLO('yolov8m-pose.pt')  # load a pretrained model (recommended for training)
    model = YOLO('yolov8x-pose-p6.pt')  # load a pretrained model (recommended for training)

    base_dir = Path("K:\\Pictures\\Photos\\2024-03-23 WuBDA Gala\\Difficult detection")
    export_dir = Path("K:\\Pictures\\Photos\\2024-03-23 WuBDA Gala\\Detection output")
    imgs = base_dir.glob("*.jpg")

    # yolo pose predict model=yolov8m-pose.pt source=''
    for img in imgs:
        start = time.time()
        result: yolo_results.Results = model.predict(img, conf=0.5)[0]  # predict on an image
        end = time.time()
        result.save(filename=Path.joinpath(export_dir, img.stem + ".jpg"))

        # keypoints = results[0].keypoints
        # image = cv2.imread(img.__str__())
        # for person in keypoints.xy:
        #     for kpt in person:
        #         x, y = int(kpt[0]), int(kpt[1])
        #         cv2.circle(image, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
        #
        # cv2.imshow("Keypoints without lines", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        print(f"Image {img.stem}: {end - start}")
        print(result.boxes.conf)
