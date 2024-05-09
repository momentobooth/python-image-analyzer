import argparse
import json
from pathlib import Path
import piexif
from PIL import Image

from momento_booth_pia.server import directory_type


def get_sources(p: Path) -> dict:
    im = Image.open(p)
    exif_dict = piexif.load(im.info["exif"])
    json_str = exif_dict["Exif"][piexif.ExifIFD.MakerNote]
    return json.loads(json_str)['sourcePhotos']


def num_pictures(p: Path) -> int:
    return len(get_sources(p))


def main():
    global args, data, data_file_path
    parser = argparse.ArgumentParser(description="MomentoBooth image creation statistics")
    parser.add_argument("-c", "--collage-dir", type=directory_type, help="Collage directory")
    args = parser.parse_args()
    print(f"Checking statistics of collage dir: {args.collage_dir}")

    collages = args.collage_dir.glob("*.jpg")
    collage_nums = [num_pictures(collage) for collage in collages]
    sums = {}
    for num in collage_nums:
        sums[num] = sums.get(num, 0) + 1

    total = sum(sums.values())
    singles = sums.get(1, 0)
    collages = total - singles
    print(f"Found {total} photos, {collages} collages, {singles} single photos")
    print(sums)


if __name__ == '__main__':
    main()
