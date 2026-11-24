"""Loading the DeepFashion dataset (Category and Attribute Prediction Benchmark).

Expected layout of ``root_dir`` (the files from the benchmark's Google Drive folder):

    list_landmarks.txt  list_bbox.txt  list_category_img.txt
    list_category_cloth.txt  list_eval_partition.txt
    img.zip   (or the unzipped img/ folder)

DeepFashion has no hosted API; ``DeepFashionAPI`` below is a small local wrapper
around this dataset so the rest of the app can ask for garments by id or search.

Landmark conventions (checked against the data):
  * ``_l`` / ``_r`` are IMAGE left / right, i.e. ``collar_l`` is the collar point
    with the smaller x. (Caleb's body keypoints use the PERSON's left/right, which
    is the mirror of this; GarmentWarper handles the mapping.)
  * The header of list_landmarks.txt lists a ``variation_type`` column that does
    not exist in the data rows. Rows are: name, clothes_type, then
    (visibility, x, y) per landmark. Visibility 0 = visible, 1 = occluded,
    2 = cut off by the image border.
"""

import os
import zipfile

import cv2
import numpy as np

UPPER, LOWER, FULL = 1, 2, 3
LANDMARK_NAMES = {
    UPPER: ["collar_l", "collar_r", "sleeve_l", "sleeve_r", "hem_l", "hem_r"],
    LOWER: ["waist_l", "waist_r", "hem_l", "hem_r"],
    FULL: ["collar_l", "collar_r", "sleeve_l", "sleeve_r",
           "waist_l", "waist_r", "hem_l", "hem_r"],
}
CLOTHES_TYPE_NAMES = {UPPER: "upper", LOWER: "lower", FULL: "full"}


def is_back_view(landmarks, clothes_type):
    """True when the photo shows the garment from behind.

    DeepFashion labels left/right from the wearer's point of view, so from the
    front "_l" points sit on the image's left; from behind they swap sides.
    """
    left, right = ("waist_l", "waist_r") if clothes_type == LOWER else ("collar_l", "collar_r")
    if left not in landmarks or right not in landmarks:
        return False
    return landmarks[left][0] > landmarks[right][0]


def _read_list(path):
    """Read a DeepFashion list file: line 1 = count, line 2 = header, then rows."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    return [line.split() for line in lines[2:] if line.strip()]


class DeepFashionDataset:
    """Reads images, categories, bounding boxes, partitions and landmarks from the local dataset."""

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.names = []          # image paths like "img/Sheer_Blouse/img_00000001.jpg"
        self.index_of = {}       # image path -> index
        self.categories = []     # [(category_name, clothes_type)], label = position + 1
        self.category_label = None
        self.bboxes = None
        self.partition = []
        self.clothes_type = None
        self.landmarks = []      # per item: {name: (x, y)}
        self.visibility = []     # per item: {name: 0/1/2}
        self._zip = None

    # ------------------------------------------------------------------ loading
    def load_annotations(self):
        """Load every annotation file. Call once before using the dataset."""
        self.load_landmarks()
        self.load_categories()
        self._load_bboxes()
        self._load_partition()
        return self

    def load_categories(self):
        cloth = os.path.join(self.root_dir, "list_category_cloth.txt")
        self.categories = [(row[0], int(row[1])) for row in _read_list(cloth)]
        rows = self._rows_by_name("list_category_img.txt")
        self.category_label = np.array([int(rows[n][1]) for n in self.names], np.int32)
        return self.categories

    def load_attributes(self):
        """Attributes are not used by the try-on pipeline (the 847 MB file was skipped)."""
        return None

    def load_landmarks(self):
        rows = _read_list(os.path.join(self.root_dir, "list_landmarks.txt"))
        self.names = [row[0] for row in rows]
        self.index_of = {n: i for i, n in enumerate(self.names)}
        self.clothes_type = np.array([int(row[1]) for row in rows], np.int8)
        self.landmarks, self.visibility = [], []
        for row, ctype in zip(rows, self.clothes_type):
            points, vis = {}, {}
            for j, name in enumerate(LANDMARK_NAMES[int(ctype)]):
                v, x, y = row[2 + 3 * j:5 + 3 * j]
                vis[name] = int(v)
                points[name] = (float(x), float(y))
            self.landmarks.append(points)
            self.visibility.append(vis)
        return self.landmarks

    def _load_bboxes(self):
        rows = self._rows_by_name("list_bbox.txt")
        self.bboxes = np.array([[int(v) for v in rows[n][1:5]] for n in self.names], np.int32)

    def _load_partition(self):
        rows = self._rows_by_name("list_eval_partition.txt")
        self.partition = [rows[n][1] for n in self.names]

    def _rows_by_name(self, filename):
        rows = {row[0]: row for row in _read_list(os.path.join(self.root_dir, filename))}
        missing = [n for n in self.names if n not in rows]
        if missing:
            raise ValueError(f"{filename} is missing {len(missing)} images, e.g. {missing[0]}")
        return rows

    # ------------------------------------------------------------------ access
    def __len__(self):
        return len(self.names)

    def read_image(self, name):
        """Return the BGR image for an image path from the annotation files."""
        folder_path = os.path.join(self.root_dir, name)
        if os.path.exists(folder_path):
            return cv2.imread(folder_path)
        if self._zip is None:
            zip_path = os.path.join(self.root_dir, "img.zip")
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"Neither {folder_path} nor {zip_path} exists")
            self._zip = zipfile.ZipFile(zip_path)
        data = np.frombuffer(self._zip.read(name), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def get_item(self, index, load_image=True):
        """Everything known about one image, as a dict."""
        label = int(self.category_label[index])
        category, _ = self.categories[label - 1]
        x1, y1, x2, y2 = (int(v) for v in self.bboxes[index])
        return {
            "id": index,
            "image_name": self.names[index],
            "image": self.read_image(self.names[index]) if load_image else None,
            "category": category,
            "category_label": label,
            "clothes_type": int(self.clothes_type[index]),
            "bbox": (x1, y1, x2, y2),
            "landmarks": dict(self.landmarks[index]),
            "visibility": dict(self.visibility[index]),
            "partition": self.partition[index],
        }

    def select(self, clothes_type=None, categories=None, partition=None, fully_visible=True,
               front_only=True):
        """Indices of items matching the filters (categories: list of names).

        front_only skips photos taken from behind, which can't be tried on.
        """
        wanted = {c.lower() for c in categories} if categories else None
        out = []
        for i in range(len(self.names)):
            if clothes_type is not None and self.clothes_type[i] != clothes_type:
                continue
            if partition is not None and self.partition[i] != partition:
                continue
            if fully_visible and any(v != 0 for v in self.visibility[i].values()):
                continue
            if front_only and is_back_view(self.landmarks[i], int(self.clothes_type[i])):
                continue
            if wanted and self.categories[self.category_label[i] - 1][0].lower() not in wanted:
                continue
            out.append(i)
        return out

    def make_garment(self, index):
        """Build a Garment for one dataset item."""
        from model.data_model import Garment
        item = self.get_item(index, load_image=False)
        return Garment(
            garment_id=index,
            image_path=item["image_name"],
            category=item["category"],
            attributes={"partition": item["partition"]},
            clothes_type=item["clothes_type"],
            bbox=item["bbox"],
            landmarks=item["landmarks"],
            visibility=item["visibility"],
            loader=self.read_image,
        )


class DeepFashionAPI:
    """Local stand-in for a "DeepFashion API": fetch and search garments.

    DeepFashion is distributed as files, not as a web service, so ``base_url`` is
    the path to the downloaded dataset folder.
    """

    def __init__(self, base_url):
        self.dataset = DeepFashionDataset(base_url).load_annotations()

    def fetch_garment(self, garment_id):
        return self.dataset.make_garment(int(garment_id))

    def search(self, query, clothes_type=None, limit=50, fully_visible=True):
        """Garment ids whose category name contains ``query`` (case-insensitive)."""
        q = query.lower()
        cats = [name for name, _ in self.dataset.categories if q in name.lower()]
        if not cats:
            return []
        return self.dataset.select(clothes_type=clothes_type, categories=cats,
                                   fully_visible=fully_visible)[:limit]
