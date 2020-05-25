import time
import os
import hashlib
import json

from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import lxml.etree
import tqdm

flags.DEFINE_string('data_dir', './data/images/', 'path to auair images')
flags.DEFINE_string('annotations', './data/annotations.json','annotations json-file')
flags.DEFINE_enum('split', 'train', [
                  'train', 'val'], 'specify train or val split')
flags.DEFINE_string('output_filebase', './data/tfrecords/auair.tfrecord', 'output dataset')
flags.DEFINE_integer('shards', '1', 'number of shards')


def build_example(annotation, class_list):
    img_path = os.path.join(
        FLAGS.data_dir, annotation['image_name'])
    print(annotation['image_name'])
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
        xmin.append(float(bbox['left']) / width)
        ymin.append(float(bbox['top']) / height)
        xmax.append( (float(bbox['left']) + float(bbox['width'])) / width)
        ymax.append( (float(bbox['top']) + float(bbox['height'])) / height)
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

    writer = tf.io.TFRecordWriter(FLAGS.output_filebase)
    for annotation in annotations["annotations"]:
        tf_example = build_example(annotation, annotations["categories"])
        writer.write(tf_example.SerializeToString())
    writer.close()
    logging.info("Done")


    # writer = tf.io.TFRecordWriter(FLAGS.output_file)
    # image_list = open(os.path.join(
    #     FLAGS.data_dir, 'ImageSets', 'Main', 'aeroplane_%s.txt' % FLAGS.split)).read().splitlines()
    # logging.info("Image list loaded: %d", len(image_list))
    # for image in tqdm.tqdm(image_list):
    #     name, _ = image.split()
    #     annotation_xml = os.path.join(
    #         FLAGS.data_dir, 'Annotations', name + '.xml')
    #     annotation_xml = lxml.etree.fromstring(open(annotation_xml).read())
    #     annotation = parse_xml(annotation_xml)['annotation']
    #     tf_example = build_example(annotation, class_map)
    #     writer.write(tf_example.SerializeToString())
    # writer.close()
    # logging.info("Done")


if __name__ == '__main__':
    app.run(main)