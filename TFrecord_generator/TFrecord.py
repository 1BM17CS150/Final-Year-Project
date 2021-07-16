import skimage.io as io
import os
from glob import glob
import tensorflow as tf


class TFrecord():
    def __init__(self):
        self.root = '.\dataset\*'
        self.tfrecords_path = './TFrecord_files/'
        if not os.path.exists(self.tfrecords_path):
            os.mkdir(self.tfrecords_path)
            print('Created Tfrecord directory')

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def generate(self):
        image_folder = glob(os.path.join(self.root))
        for folder in image_folder:
            writer = tf.python_io.TFRecordWriter(self.tfrecords_path + os.path.basename(folder) + '.tfrecords')
            img_path_list = glob(os.path.join(folder + '\*'))
            for idx, img_path in enumerate(img_path_list):
                img = io.imread(img_path)
                height = img.shape[0]
                width = img.shape[1]
                if idx % 50 == 0:
                    print('At image', idx)
                img_raw = img.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': self._int64_feature(height),
                    'width': self._int64_feature(width),
                    'image_raw': self._bytes_feature(img_raw)}))

                writer.write(example.SerializeToString())
            print('Tfrecord', os.path.basename(folder) + '.tfrecords created', end='\n\n')
            writer.close()


if __name__ == "__main__":
    TFrecord().generate()
