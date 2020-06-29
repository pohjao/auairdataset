import os
import json
import time
import math
import random
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_string('annotations', './data/annotations.json','Path to annotations json-file')
flags.DEFINE_string('output_dir', './data/yololabels', 'Output location for label files.')
flags.DEFINE_string('test_split','0.15','Portion of data used for test data.')

def to_rows(annotation):
    
    im_name = annotation['image_name']

    #TYPO in original annotations file
    im_width = int(annotation['image_width:'])
    im_height = int(annotation['image_height'])

    labels = []

    for bbox in annotation['bbox']:
        bb_left = float(bbox['left'])
        bb_top = float(bbox['top'])
        bb_width = float(bbox["width"])
        bb_height = float(bbox["height"])
        class_id = int(bbox['class'])

        rel_height = bb_height / im_height
        rel_width = bb_width / im_width
        x_center = (bb_left + bb_width) / im_width
        y_center = (bb_top + bb_height) / im_height

        if x_center > 1 or y_center > 1 or rel_height > 1 or rel_width > 1:
            print("Bounding box error.")
            continue

        row = {
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": rel_width,
            "height": rel_height
        }
        labels.append(row)
    return im_name, labels


def main(_argv):
    with open(FLAGS.annotations) as annotations_file:
        annotations = json.load(annotations_file)

    logging.info("Class mapping loaded: %s", annotations["categories"])

    file_list = annotations["annotations"]
    random.shuffle(file_list)
    first_test_index = int(math.ceil(len(file_list)*(1-float(FLAGS.test_split))))

    with open(os.path.join(FLAGS.output_dir,'obj.names'), 'w') as f:
        for item in annotations["categories"]:
            f.write("%s\n" % item)

    print("Exporting, please wait...")
    
    for idx, annotated_file in enumerate(file_list):
        im_name, labels = to_rows(annotated_file)
        if idx < first_test_index:
           file_list_filename = "train.txt"
        else:
            file_list_filename = "test.txt"
        with open(os.path.join(FLAGS.output_dir,file_list_filename), "a") as f:
            f.write("data/obj/{}\n".format(im_name))
        print("{}: {}".format(idx, im_name))
        with open(os.path.join(FLAGS.output_dir,"obj",im_name.replace(".jpg",".txt")), 'a') as f:
            for labelr in labels:
                f.write("{} {} {} {} {}\n".format(
                    labelr["class_id"],
                    labelr["x_center"],
                    labelr["y_center"],
                    labelr["width"],
                    labelr["height"]))


if __name__ == '__main__':
    app.run(main)