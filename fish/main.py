import cv2
import glob
import numpy as np
import os.path as osp
from argparse import ArgumentParser
from utils.compute_iou import compute_ious


def segment_fish(img):
    """
    This method should compute masks for given image
    Params:
        img (np.ndarray): input image in BGR format
    Returns:
        mask (np.ndarray): fish mask. should contain bool values
    """
    # convert to HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # color interval for masks
    light_orange = (1, 190, 150)
    dark_orange = (30, 255, 255)
    light_white = (60, 0, 200)
    dark_white = (145, 150, 255)

    # get masks
    orange_mask = cv2.inRange(img_hsv, light_orange, dark_orange)
    white_mask = cv2.inRange(img_hsv, light_white, dark_white)

    # masks union
    mask = cv2.bitwise_or(orange_mask, white_mask)

    if True: # morphological operation to enhance masks
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask.astype(bool)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--is_train", action="store_true")
    args = parser.parse_args()
    stage = 'train' if args.is_train else 'test'

    data_root = osp.join("dataset", stage, "imgs")
    img_paths = glob.glob(osp.join(data_root, "*.jpg"))
    len(img_paths)

    masks = dict()
    for path in img_paths:
        img = cv2.imread(path)
        mask = segment_fish(img)
        masks[osp.basename(path)] = mask
    print(compute_ious(masks, osp.join("dataset", stage, "masks")))
