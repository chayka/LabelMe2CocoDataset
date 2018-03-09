import json
import os

from glob import glob, iglob
import re
import base64
from shutil import rmtree
from PIL import Image
import argparse
from numpy import random

parser = argparse.ArgumentParser(description='Convert set of labelme files to cocodataset format')

parser.add_argument('dataset', metavar='DATASET_NAME', type=str,
                    help='dataset (folder) name')

parser.add_argument('-v, --val-ratio', dest='validation_ratio', type=float,
                    help='validation ration', default=0.2)

args = parser.parse_args()

# print(args)
# exit()

INPUT_DIR = "./input/"
OUTPUT_DIR = "./output/"
DATASET_DIR = OUTPUT_DIR + "{}/".format(args.dataset)
ANNOTATIONS_DIR = DATASET_DIR + "annotations/"


def empty_dir(path):
    """ empty specified dir """
    if os.path.exists(path):
        rmtree(path)
    os.mkdir(path)


def ensure_dir(path):
    """ empty specified dir """
    if not os.path.exists(path):
        os.mkdir(path)


def get_bbox(coords):
    """ get bounding box in format [tlx, tly, w, h] """
    min_x = None
    min_y = None
    max_x = None
    max_y = None

    for [x, y] in coords:
        min_x = x if not min_x else min(x, min_x)
        min_y = y if not min_y else min(y, min_y)
        max_x = x if not max_x else max(x, max_x)
        max_y = y if not max_y else max(y, max_y)

    return [min_x, min_y, max_x - min_x, max_y - min_y]


file_pattern = re.compile('([^/]*)\.([^/.]+)$')
category_pattern = re.compile('panel', re.IGNORECASE)

imageId = 0
annId = 0
categoryId = 0

ensure_dir(INPUT_DIR)
ensure_dir(OUTPUT_DIR)
empty_dir(DATASET_DIR)
ensure_dir(DATASET_DIR + "val")
ensure_dir(DATASET_DIR + "train")
ensure_dir(ANNOTATIONS_DIR)

images_train = []
images_val = []
annotations_train = []
annotations_val = []
categories = {}

""" Browse through all marked json files """
for file in iglob(INPUT_DIR + '{}/*.json'.format(args.dataset)):
    imageId += 1
    with open(file, 'r') as f:

        """ Load json files """
        data = json.load(f)

        """ Separation of train/validation subsets """
        subset = "val" if random.random() < args.validation_ratio else "train"

        """ Save image file """
        file_name = "{}{}/{:08d}.jpg".format(DATASET_DIR, subset, imageId)
        print(file_name, file)
        image_data = base64.b64decode(data["imageData"])
        with open(file_name, 'wb') as fi:
            fi.write(image_data)

        """ Get image width x height """
        im = Image.open(file_name)
        (width, height) = im.size

        """ Save image data to index """
        image_obj = {
            'id': imageId,
            'file_name': "{:08d}.jpg".format(imageId),
            'width': width,
            'height': height,
        }

        if subset == "val":
            images_val.append(image_obj)
        else:
            images_train.append(image_obj)

        """ Process each shape (annotation) """
        for shape in data['shapes']:
            annId += 1
            cat = shape['label']

            """ Build category index """
            if cat not in categories:
                categoryId += 1
                categories[cat] = {
                    'id': categoryId,
                    'name': cat,
                    'supercategory': 'solar panel' if category_pattern.search(cat) else 'defect'
                }
            category = categories[cat]

            """ Form segment out of points """
            segment = []
            for [x, y] in shape['points']:
                segment.append(x)
                segment.append(y)

            bbox = get_bbox(shape['points'])
            [_, _, width, height] = bbox
            """ Add annotations """
            annotation_obj = {
                'id': annId,
                'image_id': imageId,
                'category_id': category['id'],
                'segmentation': [segment],
                'bbox': bbox,
                'area': width * height,
                'iscrowd': 0,
            }

            if subset == "val":
                annotations_val.append(annotation_obj)
            else:
                annotations_train.append(annotation_obj)


with open(ANNOTATIONS_DIR + 'instances_{}_val.json'.format(args.dataset), 'w') as fa:
    json.dump({
        'images': images_val,
        'annotations': annotations_val,
        'categories': list(categories.values())
    }, fa, indent='  ')

with open(ANNOTATIONS_DIR + 'instances_{}_train.json'.format(args.dataset), 'w') as fa:
    json.dump({
        'images': images_train,
        'annotations': annotations_train,
        'categories': list(categories.values())
    }, fa, indent='  ')

