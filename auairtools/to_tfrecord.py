import time
import os
import hashlib
import json

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm
import random

flags.DEFINE_string('data_dir', './data/images/', 'Path to auair images directory')
flags.DEFINE_string('annotations', './data/annotations.json','Path to annotations json-file')
flags.DEFINE_list('splits', '80,10', 'List of split percentages: train,val. Rest of the data is split into the test set.')
flags.DEFINE_string('output_dir', './data/tfrecords', 'Output location.')

def build_example(annotation, class_list):
    img_path = os.path.join(
        FLAGS.data_dir, annotation['image_name'])
    #print(annotation['image_name'])
    img_raw = open(img_path, 'rb').read()
    key = hashlib.sha256(img_raw).hexdigest()

    #TYPO in original annotations file
    width = int(annotation['image_width:'])
    height = int(annotation['image_height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []

    for bbox in annotation['bbox']:
        bbox_xmin = float(bbox['left']) / width
        bbox_ymin = float(bbox['top']) / height
        bbox_xmax = (float(bbox['left']) + float(bbox['width'])) / width
        bbox_ymax = (float(bbox['top']) + float(bbox['height'])) / height

        if bbox_xmin >= bbox_xmax:
            print("Bounding box error in %s x_min >= x_max. Dropping bounding box." % img_path)
            continue

        if bbox_ymin >= bbox_ymax:
            print("Bounding box error in %s y_min >= y_max. Dropping bounding box." % img_path)
            continue

        xmin.append(bbox_xmin)
        ymin.append(bbox_ymin)
        xmax.append(bbox_xmax)
        ymax.append(bbox_ymax)
        classes_text.append(class_list[int(bbox['class'])].encode('utf8'))
        classes.append(int(bbox['class']))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['image_name'].encode('utf8')])),
        'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[
            annotation['image_name'].encode('utf8')])),
        'image/key/sha256': tf.train.Feature(bytes_list=tf.train.BytesList(value=[key.encode('utf8')])),
        'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
        'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmin)),
        'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmax)),
        'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymin)),
        'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymax)),
        'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
        'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
    }))

    return example

def main(_argv):
    with open(FLAGS.annotations) as annotations_file:
        annotations = json.load(annotations_file)

    logging.info("Class mapping loaded: %s", annotations["categories"])

    file_list = annotations["annotations"]
    file_count = len(file_list)
    random.shuffle(file_list)
    splits = list(map(int,FLAGS.splits))

    training_set = file_list[:int(file_count*(splits[0]/100))]
    validation_set = file_list[len(training_set):len(training_set)+int(file_count*(splits[1]/100))]
    test_set = file_list[len(validation_set)+len(training_set):]

    with open(os.path.join(FLAGS.output_dir,'auair.names'), 'w') as f:
        for item in annotations["categories"]:
            f.write("%s\n" % item)

    print("Exporting, please wait...")
    writer = tf.io.TFRecordWriter(os.path.join(FLAGS.output_dir,"auair_train.tfrecord"))
    for annotation in training_set:
        tf_example = build_example(annotation, annotations["categories"])
        writer.write(tf_example.SerializeToString())
    writer.close()


    writer = tf.io.TFRecordWriter(os.path.join(FLAGS.output_dir,"auai_validate.tfrecord"))
    for annotation in validation_set:
        tf_example = build_example(annotation, annotations["categories"])
        writer.write(tf_example.SerializeToString())
    writer.close()


    writer = tf.io.TFRecordWriter(os.path.join(FLAGS.output_dir,"auair_test.tfrecord"))
    for annotation in test_set:
        tf_example = build_example(annotation, annotations["categories"])
        writer.write(tf_example.SerializeToString())
    writer.close()

    print("Training set size: %s" % len(training_set))
    print("Validation set size: %s" % len(validation_set))
    print("Test set size: %s" % len(test_set))

    logging.info("Done")


if __name__ == '__main__':
    app.run(main)