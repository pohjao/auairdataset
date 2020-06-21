import os
import json

from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_string('annotations', './data/annotations.json','Path to annotations json-file')
flags.DEFINE_string('output_dir', './data/yololabels', 'Output location for label files.')

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
    file_count = len(file_list)

    with open(os.path.join(FLAGS.output_dir,'auair.names'), 'w') as f:
        for item in annotations["categories"]:
            f.write("%s\n" % item)

    print("Exporting, please wait...")
    
    for annotated_file in file_list:
        im_name, labels = to_rows(annotated_file)
        print(im_name)
        with open(os.path.join(FLAGS.output_dir,im_name.replace(".jpg",".txt")), 'a') as f:
            for labelr in labels:
                f.write("{} {} {} {} {}\n".format(
                    labelr["class_id"],
                    labelr["x_center"],
                    labelr["y_center"],
                    labelr["width"],
                    labelr["height"]
                ))



    logging.info("Done")


if __name__ == '__main__':
    app.run(main)