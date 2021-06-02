import argparse
import mnist
import pathlib
import cv2
import numpy as np
import tqdm


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.
    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = prediction_box
    if (x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t):
        return 0.0

    # Compute intersection
    x1i = max(x1_t, x1_p)
    x2i = min(x2_t, x2_p)
    y1i = max(y1_t, y1_p)
    y2i = min(y2_t, y2_p)
    intersection = (x2i - x1i) * (y2i - y1i)

    # Compute union
    pred_area = (x2_p - x1_p) * (y2_p - y1_p)
    gt_area = (x2_t - x1_t) * (y2_t - y1_t)
    union = pred_area + gt_area - intersection
    iou = intersection / union
    assert iou >= 0 and iou <= 1
    return iou


def compute_iou_all(bbox, all_bboxes):
    ious = [0]
    for other_bbox in all_bboxes:
        ious.append(
            calculate_iou(bbox, other_bbox)
        )
    return ious


def tight_bbox(digit, orig_bbox):
    xmin, ymin, xmax, ymax = orig_bbox
    # xmin
    shift = 0
    for i in range(digit.shape[1]):
        if digit[:, i].sum() != 0:
            break
        shift += 1
    xmin += shift
    # xmax
    shift = 0
    for i in range(-1, -digit.shape[1], -1):
        if digit[:, i].sum() != 0:
            break
        shift += 1
    xmax -= shift
    ymin
    shift = 0
    for i in range(digit.shape[0]):
        if digit[i, :].sum() != 0:
            break
        shift += 1
    ymin += shift
    shift = 0
    for i in range(-1, -digit.shape[0], -1):
        if digit[i, :].sum() != 0:
            break
        shift += 1
    ymax -= shift
    return [xmin, ymin, xmax, ymax]


def dataset_exists(dirpath: pathlib.Path, num_images):
    if not dirpath.is_dir():
        return False
    for image_id in range(num_images):
        error_msg = f"MNIST dataset already generated in {dirpath}, \n\tbut did not find filepath:"
        error_msg2 = f"You can delete the directory by running: rm -r {dirpath.parent}"
        impath = dirpath.joinpath("images", f"{image_id}.png")
        assert impath.is_file(), f"{error_msg} {impath} \n\t{error_msg2}"
        label_path = dirpath.joinpath("labels", f"{image_id}.txt")
        assert label_path.is_file(),  f"{error_msg} {impath} \n\t{error_msg2}"
    return True


def generate_dataset(dirpath: pathlib.Path,
                     num_images: int,
                     max_digit_size: int,
                     min_digit_size: int,
                     imsize: int,
                     max_digits_per_image: int,
                     mnist_images: np.ndarray,
                     mnist_labels: np.ndarray):
    if dataset_exists(dirpath, num_images):
        return
    max_image_value = 255
    assert mnist_images.dtype == np.uint8
    image_dir = dirpath.joinpath("images")
    label_dir = dirpath.joinpath("labels")
    image_dir.mkdir(exist_ok=True, parents=True)
    label_dir.mkdir(exist_ok=True, parents=True)
    for image_id in tqdm.trange(num_images, desc=f"Generating dataset, saving to: {dirpath}"):
        im = np.zeros((imsize, imsize), dtype=np.float32)
        labels = []
        bboxes = []
        num_images = np.random.randint(0, max_digits_per_image)
        for _ in range(num_images+1):
            while True:
                width = np.random.randint(min_digit_size, max_digit_size)
                x0 = np.random.randint(0, imsize-width)
                y0 = np.random.randint(0, imsize-width)
                ious = compute_iou_all([x0, y0, x0+width, y0+width], bboxes)
                if max(ious) < 0.25:
                    break
            digit_idx = np.random.randint(0, len(mnist_images))
            digit = mnist_images[digit_idx].astype(np.float32)
            digit = cv2.resize(digit, (width, width))
            label = mnist_labels[digit_idx]
            labels.append(label)
            assert im[y0:y0+width, x0:x0+width].shape == digit.shape, \
                f"imshape: {im[y0:y0+width, x0:x0+width].shape}, digit shape: {digit.shape}"
            bbox = tight_bbox(digit, [x0, y0, x0+width, y0+width])
            bboxes.append(bbox)

            im[y0:y0+width, x0:x0+width] += digit
            im[im > max_image_value] = max_image_value
        image_target_path = image_dir.joinpath(f"{image_id}.png")
        label_target_path = label_dir.joinpath(f"{image_id}.txt")
        im = im.astype(np.uint8)
        cv2.imwrite(str(image_target_path), im)
        with open(label_target_path, "w") as fp:
            fp.write("label,xmin,ymin,xmax,ymax\n")
            for l, bbox in zip(labels, bboxes):
                bbox = [str(_) for _ in bbox]
                to_write = f"{l}," + ",".join(bbox) + "\n"
                fp.write(to_write)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-path", default="data/mnist_detection"
    )
    parser.add_argument(
        "--imsize", default=300, type=int
    )
    parser.add_argument(
        "--max-digit-size", default=100, type=int
    )
    parser.add_argument(
        "--min-digit-size", default=15, type=int
    )
    parser.add_argument(
        "--num-train-images", default=10000, type=int
    )
    parser.add_argument(
        "--num-test-images", default=1000, type=int
    )
    parser.add_argument(
        "--max-digits-per-image", default=20, type=int
    )
    args = parser.parse_args()
    X_train, Y_train, X_test, Y_test = mnist.load()
    for dataset, (X, Y) in zip(["train", "test"], [[X_train, Y_train], [X_test, Y_test]]):
        num_images = args.num_train_images if dataset == "train" else args.num_test_images
        generate_dataset(
            pathlib.Path(args.base_path, dataset),
            num_images,
            args.max_digit_size,
            args.min_digit_size,
            args.imsize,
            args.max_digits_per_image,
            X,
            Y) 
