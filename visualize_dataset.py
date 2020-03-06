import matplotlib.pyplot as plt
import argparse
import pathlib
import numpy as np
import typing

colors = ["blue", "green", "cyan", "red", "yellow", "magenta", "peru", "azure", "slateblue", "plum"]


def plot_bbox(bbox_XYXY, label):
    xmin, ymin, xmax, ymax =bbox_XYXY
    plt.plot(
        [xmin, xmin, xmax, xmax, xmin],
        [ymin, ymax, ymax, ymin, ymin],
        color=colors[label], 
        label=str(label))

def read_labels(label_path: pathlib.Path) -> typing.Tuple[np.ndarray]:
    assert label_path.is_file()
    labels = []
    BBOXES_XYXY = []
    with open(label_path, "r") as fp:
        for line in list(fp.readlines())[1:]:
            label, xmin, ymin, xmax, ymax = [int(_) for _ in line.split(",")]
            labels.append(label)
            BBOXES_XYXY.append([xmin, ymin, xmax, ymax])
    return np.array(labels), np.array(BBOXES_XYXY)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    args = parser.parse_args()

    base_path = pathlib.Path(args.directory)
    image_dir = base_path.joinpath("images")
    label_dir = base_path.joinpath("labels")
    impaths = image_dir.glob("*.png")
    for impath in impaths:
        label_path = label_dir.joinpath(
            f"{impath.stem}.txt"
        )
        labels, bboxes_XYXY = read_labels(label_path)
        im = plt.imread(str(impath))
        plt.imshow(im, cmap="gray")
        for bbox, label in zip(bboxes_XYXY, labels):
            plot_bbox(bbox, label)
        plt.savefig("example_image.png")
        plt.show()
