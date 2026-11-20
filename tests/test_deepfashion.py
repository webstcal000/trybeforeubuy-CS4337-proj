"""DeepFashion Loader Tests! Uses a tiny fake dataset in the real file format."""

import os
import sys
import zipfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.deepfashion import DeepFashionAPI, DeepFashionDataset  # noqa: E402

NAMES = ["img/Red_Tee/img_00000001.jpg", "img/Blue_Jeans/img_00000001.jpg",
         "img/Red_Tee/img_00000002.jpg"]


def write_fake_dataset(root, zipped=False):
    # Same layout as the real files, including the bogus "variation_type" header column.
    with open(os.path.join(root, "list_landmarks.txt"), "w") as f:
        f.write("3\nimage_name  clothes_type  variation_type  landmark_visibility_1 ...\n")
        f.write(f"{NAMES[0]}   1  0 046 030  0 074 030  0 020 060  0 100 060  0 040 110  0 080 112\n")
        f.write(f"{NAMES[1]}   2  0 040 010  0 080 010  0 030 115  0 090 115\n")
        f.write(f"{NAMES[2]}   1  0 046 030  1 074 030  0 020 060  0 100 060  2 040 110  0 080 112\n")
    with open(os.path.join(root, "list_category_cloth.txt"), "w") as f:
        f.write("3\ncategory_name  category_type\nTee   1\nJeans   2\nDress   3\n")
    with open(os.path.join(root, "list_category_img.txt"), "w") as f:
        f.write(f"3\nimage_name  category_label\n{NAMES[0]}  1\n{NAMES[1]}  2\n{NAMES[2]}  1\n")
    with open(os.path.join(root, "list_bbox.txt"), "w") as f:
        f.write(f"3\nimage_name  x_1  y_1  x_2  y_2\n{NAMES[0]}  010 020 110 118\n"
                f"{NAMES[1]}  020 005 100 118\n{NAMES[2]}  010 020 110 118\n")
    with open(os.path.join(root, "list_eval_partition.txt"), "w") as f:
        f.write(f"3\nimage_name  evaluation_status\n{NAMES[0]}  train\n{NAMES[1]}  test\n{NAMES[2]}  val\n")
    image = np.full((120, 120, 3), 128, np.uint8)
    ok, jpg = cv2.imencode(".jpg", image)
    if zipped:
        with zipfile.ZipFile(os.path.join(root, "img.zip"), "w") as z:
            for n in NAMES:
                z.writestr(n, jpg.tobytes())
    else:
        for n in NAMES:
            os.makedirs(os.path.dirname(os.path.join(root, n)), exist_ok=True)
            cv2.imwrite(os.path.join(root, n), image)


def test_parses_landmarks_despite_bad_header(tmp_path):
    write_fake_dataset(tmp_path)
    ds = DeepFashionDataset(str(tmp_path)).load_annotations()
    assert len(ds) == 3
    item = ds.get_item(0)
    assert item["clothes_type"] == 1 and item["category"] == "Tee"
    assert item["landmarks"]["collar_l"] == (46.0, 30.0)
    assert item["landmarks"]["hem_r"] == (80.0, 112.0)
    assert item["bbox"] == (10, 20, 110, 118) and item["partition"] == "train"
    assert item["image"].shape == (120, 120, 3)
    lower = ds.get_item(1, load_image=False)
    assert set(lower["landmarks"]) == {"waist_l", "waist_r", "hem_l", "hem_r"}


def test_reads_images_from_zip(tmp_path):
    write_fake_dataset(tmp_path, zipped=True)
    ds = DeepFashionDataset(str(tmp_path)).load_annotations()
    assert ds.get_item(2)["image"].shape == (120, 120, 3)


def test_select_and_garment(tmp_path):
    write_fake_dataset(tmp_path)
    ds = DeepFashionDataset(str(tmp_path)).load_annotations()
    assert ds.select(clothes_type=1) == [0]                    # item 2 has hidden landmarks
    assert ds.select(clothes_type=1, fully_visible=False) == [0, 2]
    assert ds.select(partition="test") == [1]
    g = ds.make_garment(2)
    assert set(g.get_landmarks()) == {"collar_l", "sleeve_l", "sleeve_r", "hem_r"}
    assert g.load().shape == (120, 120, 3)


def test_api_search(tmp_path):
    write_fake_dataset(tmp_path)
    api = DeepFashionAPI(str(tmp_path))
    assert api.search("tee") == [0]
    assert api.search("jean") == [1]
    assert api.fetch_garment(1).category == "Jeans"
