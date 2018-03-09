import json
import os

from glob import glob, iglob
import re
import base64
from shutil import rmtree
from PIL import Image

DATA_DIR = "./data/"
IMAGES_DIR = DATA_DIR + "solar_panels/"
ANNOTATIONS_DIR = DATA_DIR + "annotations/"


def empty_dir(path):
    if os.path.exists(path):
        rmtree(path)
    os.mkdir(path)


def get_bbox(coords):
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

empty_dir(IMAGES_DIR)
empty_dir(ANNOTATIONS_DIR)

images = []
annotations = []
categories = {}
""" Browse through all marked json files """
for file in iglob(DATA_DIR + '*.json'):
    imageId += 1
    with open(file, 'r') as f:

        """ Load json files """
        data = json.load(f)

        """ Save image file """
        file_name = "{}{:08d}.jpg".format(IMAGES_DIR, imageId)
        print(file_name, file)
        if not os.path.isfile(file_name):
            imageData = base64.b64decode(data["imageData"])
            with open(file_name, 'wb') as fi:
                fi.write(imageData)
        """ Get image width x height """
        im = Image.open(file_name)
        (width, height) = im.size

        """ Save image data to index """
        images.append({
            'id': imageId,
            'file_name': "{:08d}.jpg".format(imageId),
            'width': width,
            'height': height,
        })

        # del data["imageData"]
        # print(data)

        """ Process each shape (annotation) """
        for shape in data['shapes']:
            annId += 1
            # print(shape)
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

            """ Add annotations """
            annotations.append({
                'id': annId,
                'image_id': imageId,
                'category_id': category['id'],
                'segmentation': [segment],
                'bbox': get_bbox(shape['points']),
                'iscrowd': 0,
            })

output = {
    'images': images,
    'annotations': annotations,
    'categories': list(categories.values())
}

with open(ANNOTATIONS_DIR + 'instances_solar_panels.json', 'w') as fa:
    json.dump(output, fa, indent='  ')
