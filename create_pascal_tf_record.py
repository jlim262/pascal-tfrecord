# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Convert raw PASCAL VOC format dataset to TFRecord for object_detection.

Example usage:
    ./create_pascal_tf_record \
        --label_map_path /home/data/label_map.pbtxt \
        --annotations_dirs /home/data/labelled1 /home/data/labelled2 \
        --image_dirs /home/data/image1 /home/data/image1
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import hashlib
import io
import logging
import os
import random
from lxml import etree

import PIL.Image
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

logging.basicConfig(level=logging.INFO)

flags = tf.app.flags
flags.DEFINE_string('label_map_path', '', 'Path to label map proto')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('train_output_path', 'train.record', 'File path to output TFRecord for training')
flags.DEFINE_string('val_output_path', 'val.record', 'File path to output TFRecord for evaluation')
flags.DEFINE_string('original_image_format', 'jpeg', 'Original image format')
flags.DEFINE_string('converted_image_format', 'jpeg', 'Converted image format')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore difficult instances')
flags.DEFINE_boolean('convert_image', True, 'Whether to convert image format')
FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       label_map_dict,
                       ignore_difficult_instances=False):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
      data: dict holding PASCAL XML fields for a single image.
      label_map_dict: A map from string label names to integers ids.
      ignore_difficult_instances: Whether to skip difficult instances in the dataset  (default: False).

    Returns:
      example: The converted tf.Example.

    Raises:
      ValueError: if the image pointed to by data['filename'] is not a valid format
    """

    if FLAGS.convert_image:
        img_type = FLAGS.converted_image_format
    else:
        img_type = FLAGS.original_image_format

    image_fullpath = data['image_fullpath']
    if not os.path.exists(image_fullpath):
        raise IOError

    with tf.gfile.GFile(image_fullpath, 'rb') as fid:
        try:
            encoded_file = fid.read()
        except IOError:
            raise

    encoded_file_io = io.BytesIO(encoded_file)
    image = PIL.Image.open(encoded_file_io)
    if image.format.upper() != img_type.upper():
        raise ValueError('Image format not ' + img_type)
    key = hashlib.sha256(encoded_file).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []
    for obj in data['object']:
        difficult = bool(int(obj['difficult']))
        if ignore_difficult_instances and difficult:
            continue

        difficult_obj.append(int(difficult))

        xmin.append(float(obj['bndbox']['xmin']) / width)
        ymin.append(float(obj['bndbox']['ymin']) / height)
        xmax.append(float(obj['bndbox']['xmax']) / width)
        ymax.append(float(obj['bndbox']['ymax']) / height)
        classes_text.append(obj['name'].encode('utf8'))
        classes.append(label_map_dict[obj['name']])
        truncated.append(int(obj['truncated']))
        poses.append(obj['pose'].encode('utf8'))

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_file),
        'image/format': dataset_util.bytes_feature(img_type.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return example


def create_tf_record(output_path,
                     label_map_dict,
                     dataset):
    writer = tf.python_io.TFRecordWriter(output_path)
    save_count = 0
    for image_fullpath, annotation_file_path in dataset:
        with tf.gfile.GFile(annotation_file_path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        data['image_fullpath'] = image_fullpath
        try:
            tf_example = dict_to_tf_example(data, label_map_dict,
                                            FLAGS.ignore_difficult_instances)
        except IOError:
            print('ignore the missing file.')
            continue

        writer.write(tf_example.SerializeToString())
        save_count += 1

    writer.close()
    logging.info('{} dataset is written to {}.'.format(save_count, output_path))


def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dirs', nargs='+', type=str)
    parser.add_argument('--annotations_dirs', nargs='+', type=str)
    args = parser.parse_known_args()
    image_dirs = args[0].image_dirs
    annotations_dirs = args[0].annotations_dirs

    if len(image_dirs) != len(annotations_dirs):
        logging.info('The number of --image_dirs({}) and --annotation_dirs({}) should be same.'.
                     format(len(image_dirs), len(annotations_dirs)))
        return

    dataset_list = []
    for image_dir, annotation_dir in zip(image_dirs, annotations_dirs):
        for f in os.listdir(annotation_dir):
            if f.endswith('.xml'):
                image_filename = os.path.splitext(f)[0] + "." + FLAGS.original_image_format
                image_fullpath = os.path.join(image_dir, image_filename)
                if os.path.exists(image_fullpath):
                    if FLAGS.convert_image:
                        converted_image_filename = os.path.splitext(f)[0] + "." + FLAGS.converted_image_format
                        converted_image_fullpath = os.path.join(image_dir, converted_image_filename)
                        if not os.path.exists(converted_image_fullpath):
                            img = PIL.Image.open(image_fullpath)
                            img.save(converted_image_fullpath, FLAGS.converted_image_format)
                        dataset_list.append((converted_image_fullpath, os.path.join(annotation_dir, f)))
                    else:
                        if os.path.exists(image_fullpath):
                            dataset_list.append((image_fullpath, os.path.join(annotation_dir, f)))

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    random.seed(42)
    random.shuffle(dataset_list)
    num_dataset = len(dataset_list)
    num_train = int(0.7 * num_dataset)

    train_dataset = dataset_list[:num_train]
    val_dataset = dataset_list[num_train:]
    logging.info('{} training and {} validation examples.'.format(len(train_dataset), len(val_dataset)))

    train_output_path = os.path.join(FLAGS.output_dir, FLAGS.train_output_path)
    val_output_path = os.path.join(FLAGS.output_dir, FLAGS.val_output_path)

    create_tf_record(train_output_path, label_map_dict, train_dataset)
    create_tf_record(val_output_path, label_map_dict, val_dataset)


if __name__ == '__main__':
    tf.app.run()
