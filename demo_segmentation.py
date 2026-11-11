
"""Body Segmentation Demo!"""

import argparse
import os
import time

import cv2

from controller.body_segmentation import BodySegmenter, draw_debug


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image")
    parser.add_argument("--out", help="where to save the overlay (default: <image>_seg.png)")
    parser.add_argument("--show", action="store_true", help="open a window with the result")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read {args.image}")

    segmenter = BodySegmenter()
    start = time.perf_counter()
    result = segmenter.run(image)
    elapsed = time.perf_counter() - start

    print(f"bbox: {result['bbox']}  keypoints from: {result['keypoint_source']}  ({elapsed:.2f}s)")
    for name, point in result["keypoints"].items():
        print(f"  {name:15s} {point}")

    overlay = draw_debug(image, result)
    out = args.out or os.path.splitext(args.image)[0] + "_seg.png"
    cv2.imwrite(out, overlay)
    cv2.imwrite(os.path.splitext(out)[0] + "_mask.png", result["body_mask"])
    print(f"saved {out}")

    if args.show:
        cv2.imshow("segmentation", overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
