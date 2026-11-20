"""Try-On Warping Demo!

Examples
  python demo_warping.py --dataset path/to/deepfashion --person me.jpg --garment 1234
  python demo_warping.py --dataset path/to/deepfashion --person me.jpg --search tee --count 4
  python demo_warping.py --dataset path/to/deepfashion --person-id 77 --search blouse

Saves one strip per garment: photo | garment cut-out | affine | perspective | thin plate spline.
"""

import argparse
import os
import random
import time

import cv2
import numpy as np

from controller.body_segmentation import BodySegmenter
from controller.image_warping import METHODS, GarmentWarper
from evaluation.evaluate import garment_fit_scores
from model.deepfashion import DeepFashionAPI


def label(image, text):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fit_height(image, h):
    return cv2.resize(image, (int(image.shape[1] * h / image.shape[0]), h))


def garment_preview(garment):
    image, mask = garment.load(), garment.get_mask()
    out = np.full_like(image, 255)
    out[mask > 0] = image[mask > 0]
    for (x, y) in garment.get_landmarks().values():
        cv2.circle(out, (int(x), int(y)), 3, (0, 200, 0), -1)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True, help="folder with the DeepFashion list_*.txt files and img.zip")
    who = p.add_mutually_exclusive_group(required=True)
    who.add_argument("--person", help="photo of the person to dress")
    who.add_argument("--person-id", type=int, help="use a DeepFashion image as the person (for testing)")
    p.add_argument("--garment", type=int, nargs="*", help="garment ids")
    p.add_argument("--search", help="category name to pick garments from, e.g. tee, blouse, dress")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="tryon_results")
    args = p.parse_args()

    api = DeepFashionAPI(args.dataset)
    ids = list(args.garment or [])
    if args.search:
        found = api.search(args.search, clothes_type=None)
        random.Random(args.seed).shuffle(found)
        ids += found[:args.count]
    if not ids:
        raise SystemExit("no garments chosen: use --garment or --search")

    person = cv2.imread(args.person) if args.person else api.dataset.get_item(args.person_id)["image"]
    if person is None:
        raise SystemExit(f"could not read {args.person}")
    if person.shape[0] < 512:   # DeepFashion photos are small; work at a usable size
        person = fit_height(person, 512)

    seg = BodySegmenter().run(person)
    print("keypoints from", seg["keypoint_source"])
    os.makedirs(args.out, exist_ok=True)
    warper = GarmentWarper()
    h = person.shape[0]

    for gid in ids:
        garment = api.fetch_garment(gid)
        tiles = [label(person, "photo"), label(fit_height(garment_preview(garment), h), garment.category)]
        for method in METHODS:
            start = time.perf_counter()
            try:
                result = warper.try_on(person, seg, garment, method=method)
            except ValueError as e:
                print(f"garment {gid}: {e}")
                break
            ms = 1000 * (time.perf_counter() - start)
            scores = garment_fit_scores(warper.last, seg)
            print(f"garment {gid} {method:11s} {ms:6.0f} ms  "
                  f"landmark error {scores['landmark_error']:.3f}  torso covered {scores['torso_coverage']:.2f}")
            tiles.append(label(result, method))
        else:
            path = os.path.join(args.out, f"garment_{gid}.jpg")
            cv2.imwrite(path, np.hstack(tiles))
            print("saved", path)


if __name__ == "__main__":
    main()
