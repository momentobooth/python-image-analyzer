import time
from pathlib import Path

import face_recognition
import numpy as np
from PIL import Image
import piexif
import json


def process_collage(source: Path) -> (dict, None):
    pillow_collage_img = Image.open(source)
    print(f"Opening image {source.stem}")
    exif_dict = piexif.load(pillow_collage_img.info["exif"])
    json_str = exif_dict["Exif"][piexif.ExifIFD.MakerNote]
    json_obj = json.loads(json_str)
    print(f"\tJSON object: {json_obj}")
    sources = [next(source_dir.glob(f"*{x['filename']}"), None) for x in json_obj["sourcePhotos"]]
    sources = [source for source in sources if source is not None]
    if len(sources) == 0:
        print("\tCould not find source files, skipping")
        return None
    else:
        print(f"\tImage has {len(sources)} source(s)")
    last_source = sources[-1]
    pass


if __name__ == "__main__":
    output_img = Path("K:\\Pictures\\Photos\\2024-03-23 WuBDA Gala\\Output\\MomentoBooth-image-0097.jpg")
    source_dir = Path("K:\\Pictures\\Photos\\2024-03-23 WuBDA Gala\\From camera")

    process_result = process_collage(output_img)
    print("Completed analysis!")
