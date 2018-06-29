import sys
import pandas as pd
import os
import numpy as np
import cv2
import time


def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def label_to_heatmap(label, image):
    """
    generator heatmaps from annotations
    :param label: list
    :param image: ndarray (512, 512, 3)
    :return: heatmap (512, 512, 24)
    """
    height, width, _ = image.shape
    heatmap = np.zeros((height, width, 24))
    # radius
    r = 20
    # meshgrid
    meshgrid = np.meshgrid(range(width), range(height))

    num_visible = 0

    # label to heatmaps
    for i in range(24):
        if label[i * 3 + 2] == 1:
            num_visible += 1
            # center
            c = label[i * 3:i * 3 + 2]
            inds = ((meshgrid[0] - c[0]) ** 2 + (meshgrid[1] - c[1]) ** 2) <= r * r
            heatmap[inds, i] = 1
    heatmap = heatmap.astype(np.float32)
    return heatmap, num_visible


def heatmap_to_label(pred_heatmap, size):
    """
    generate labels according predicted heatmaps
    :param pred_heatmap: ndarray (512, 512, 24)
    :param size: tuple, (height, width)
    :return:
    """
    pred_heatmap = cv2.GaussianBlur(pred_heatmap, (41, 41), 0)
    keypoints = []
    for i in range(24):
        temp = pred_heatmap[:, :, i]
        argm = np.argmax(temp)
        y, x = np.unravel_index(argm, size)
        keypoint = (x, y)
        keypoints.append(keypoint)

    return keypoints


def prepare_test_paths(annotations_dir, test_dir):
    """
    read annotations file
    :param annotations_dir: str
    :param test_dir: str
    :return:
        image_paths: list
        labels: list
    """
    print("preparing testing data")
    raw_data = pd.read_csv(annotations_dir)

    image_paths = []
    labels = []

    for i in range(raw_data.shape[0]):
        print_progress(count=i, total=raw_data.shape[0] - 1)
        line = raw_data.iloc[i]

        # read image
        try:
            image_path = os.path.join(test_dir, line["image_id"])
            image_paths.append(image_path)
        except IOError:
            print("read image error")

        # read label
        label = []
        for _ in range(24):
            label.append(0)
            label.append(0)
            label.append(1)
        labels.append(label)
    return image_paths, labels


def prepare_data(data_dir, is_shuffle=True):
    """
    read annotations file
    :param data_dir: str
    :param is_shuffle: boolean
    :return:
        image_paths: list
        labels: list
    """
    print("preparing training data")
    annotations_dir = 'Annotations/annotations.csv'
    raw_data = pd.read_csv(os.path.join(data_dir, annotations_dir))
    # shuffle
    if is_shuffle:
        raw_data = raw_data.sample(frac=1)
    image_paths = []
    labels = []

    for i in range(raw_data.shape[0]):
        print_progress(count=i, total=raw_data.shape[0] - 1)
        line = raw_data.iloc[i]

        # read image
        try:
            image_path = os.path.join(data_dir, line["image_id"])
            image_paths.append(image_path)
        except IOError:
            print("read image error")

        # read label
        label = []
        for keypoint in line[2:]:
            x, y, v = keypoint.split('_')
            label.append(int(x))
            label.append(int(y))
            label.append(int(v))
        labels.append(label)
    print("...")
    print("finished")
    return image_paths, labels


def flip_to_origin(heatmap_flip):
    """
    swap left and right points
    :param heatmap_flip: ndarray (512, 512, 24)
    :return:
        valid flipped heatmap: ndarray (512, 512, 24)
    """
    heatmap_flip = heatmap_flip[:, ::-1, :]
    heatmap = np.copy(heatmap_flip)

    map_from = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 10, 12, 13, 14, 15, 16, 17, 18, 20, 22, 21, 23]
    map_to = [1, 0, 4, 3, 6, 5, 8, 7, 11, 9, 12, 10, 14, 13, 16, 15, 18, 17, 22, 20, 23, 21]

    heatmap[:, :, map_from] = heatmap_flip[:, :, map_to]

    return heatmap


def writer(output_dir, queue, stop_token='stop'):
    head = 'image_id,image_category,' \
           'neckline_left,neckline_right,' \
           'center_front,shoulder_left,shoulder_right,' \
           'armpit_left,armpit_right,' \
           'waistline_left,waistline_right,' \
           'cuff_left_in,cuff_left_out,' \
           'cuff_right_in,cuff_right_out,' \
           'top_hem_left,top_hem_right,' \
           'waistband_left,waistband_right,' \
           'hemline_left,hemline_right,' \
           'crotch, bottom_left_in,bottom_left_out,' \
           'bottom_right_in,bottom_right_out\n'

    with open(output_dir, 'w') as f:
        f.write(head)
        while True:
            token, img_path, heatmaps = queue.get()
            if token == stop_token:
                return
            heatmap = heatmaps[0]
            heatmap_flip = heatmaps[1]
            heatmap_2 = flip_to_origin(heatmap_flip)
            heatmap_mixed = heatmap + heatmap_2
            img_path = img_path[0]
            size = (512, 512)
            pred = heatmap_to_label(heatmap_mixed, size)
            paths = img_path.split('/')
            cat = paths[2]
            img_path = os.path.join(*paths[1:])
            print(img_path)
            st = img_path
            st = st + ',' + cat
            for i in range(24):
                st += ',{}_{}_1'.format(pred[i][0], pred[i][1])
            st += '\n'
            f.write(st)


def name_in_checkpoint(var):
    if 'fpn1' in var.op.name:
        return var.op.name.replace('fpn1/', '')
    if 'fpn2' in var.op.name:
        return var.op.name.replace('fpn2/', '')
