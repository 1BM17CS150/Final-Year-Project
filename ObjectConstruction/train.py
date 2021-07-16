import os
import time
import tensorflow as tf
import pickle
import argparse
from tkinter import *
from tkinter.ttk import *
from glob import glob
from utils import Utils
from gan import GAN
from helping_functions import *

"""
    python3 train.py --epoch 300  --batchsize 128 --maxitr 100050 --checkpoint 100000
    python3 train.py -e 100 -b 64 -m 100050 -c 100000
"""

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', '-e', type=int, default=300)
parser.add_argument('--batchsize', '-b', type=int, default=128)
parser.add_argument('--maxitr', '-m', type=int, default=100000)
parser.add_argument('--checkpoint', '-c', type=int, default=0)
options = parser.parse_args()
data_dir = '../TFrecord_generator/TFrecord_files'
log_dir = 'checkpoints'
images_dir = 'images'
max_itr = options.maxitr
batch_size = options.batchsize
latest_ckpt = options.checkpoint
num_per_epoch = options.epoch
nb_channels = 1
do_train = True

CROP_SIZE = 96


def decode_tfrecords(data_dir, batch_size, s_size):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.tfrecords')]
    fqueue = tf.train.string_input_producer(files)
    reader = tf.TFRecordReader()
    _, serialized = reader.read(fqueue)
    features = tf.parse_single_example(serialized, features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)})

    image = tf.cast(tf.decode_raw(features['image_raw'], tf.uint8), tf.float32)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image = tf.reshape(image, [height, width, nb_channels])
    image = tf.image.resize_image_with_crop_or_pad(image, CROP_SIZE, CROP_SIZE)

    min_queue_examples = num_per_epoch
    images = tf.train.shuffle_batch(
        [image],
        batch_size=batch_size,
        capacity=min_queue_examples + nb_channels * batch_size,
        min_after_dequeue=min_queue_examples)
    tf.summary.image('images', images)
    return tf.subtract(tf.div(tf.image.resize_images(images, [s_size * 2 ** 4, s_size * 2 ** 4]), 127.5), 1.0)


def main(_):
    gan = GAN(batch_size=batch_size, s_size=6, nb_channels=nb_channels)
    print('Inside main')
    traindata = decode_tfrecords(data_dir, gan.batch_size, gan.s_size)
    losses = gan.loss(traindata)

    # feature matching
    graph = tf.get_default_graph()
    features_g = tf.reduce_mean(graph.get_tensor_by_name('dg/d/conv4/outputs:0'), 0)
    features_t = tf.reduce_mean(graph.get_tensor_by_name('dt/d/conv4/outputs:0'), 0)
    losses[gan.g] += tf.multiply(tf.nn.l2_loss(features_g - features_t), 0.05)

    tf.summary.scalar('g_loss', losses[gan.g])
    tf.summary.scalar('d_loss', losses[gan.d])
    train_op = gan.train(losses, learning_rate=0.0001)
    summary_op = tf.summary.merge_all()

    g_saver = tf.train.Saver(gan.g.variables, max_to_keep=15)
    d_saver = tf.train.Saver(gan.d.variables, max_to_keep=15)
    g_checkpoint_path = os.path.join(log_dir, 'g.ckpt')
    d_checkpoint_path = os.path.join(log_dir, 'd.ckpt')
    g_checkpoint_restore_path = os.path.join(log_dir, 'g.ckpt-' + str(latest_ckpt))
    d_checkpoint_restore_path = os.path.join(log_dir, 'd.ckpt-' + str(latest_ckpt))

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)

        sess.run(tf.global_variables_initializer())
        # restore or initialize generator
        if os.path.exists(g_checkpoint_restore_path + '.meta'):
            print('Restoring variables:')
            for v in gan.g.variables:
                print(' ' + v.name)
            g_saver.restore(sess, g_checkpoint_restore_path)
        else:
            print("Generator initialisation failed ")

        if do_train:
            print("Training Initiated")
            # restore or initialize discriminator
            if os.path.exists(d_checkpoint_restore_path + '.meta'):
                print('Restoring variables:')
                for v in gan.d.variables:
                    print(' ' + v.name)
                d_saver.restore(sess, d_checkpoint_restore_path)
            else:
                print("Discriminator initialisation failed ")

            # setup for monitoring
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            sample_z = sess.run(tf.random_uniform([gan.batch_size, gan.z_dim], minval=-1.0, maxval=1.0))
            images = gan.sample_images(5, 5, inputs=sample_z)

            filename = os.path.join(images_dir, '000000.jpg')
            with open(filename, 'wb') as f:
                f.write(sess.run(images))

            threads = tf.train.start_queue_runners(coord=coord)
            print("""Starting Iterations""")
            d_hist = []
            g_hist = []
            for itr in range(latest_ckpt + 1, max_itr + 1):
                start_time = time.time()
                _, g_loss, d_loss = sess.run([train_op, losses[gan.g], losses[gan.d]])
                duration = time.time() - start_time
                if itr % 50 == 0:
                    print('step: %d, loss: (G: %.8f, D: %.8f), time taken: %.3f' % (itr, g_loss, d_loss, duration))
                    d_hist.append(d_loss)
                    g_hist.append(g_loss)
                if itr % 500 == 0:
                    # Images generated

                    filename = os.path.join(images_dir, '%06d.jpg' % itr)
                    with open(filename, 'wb') as f:
                        f.write(sess.run(images))
                    print('Generated Image : %s' % filename)
                    # Summary
                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, itr)

                # Checkpoints
                if itr % 1000 == 0:
                    print('Creating Checkpoint')
                    g_saver.save(sess, g_checkpoint_path, global_step=itr)
                    d_saver.save(sess, d_checkpoint_path, global_step=itr)
            print("""Iterations Ended""")
            coord.request_stop()
            coord.join(threads)
            with open('losses.pickle', 'ab') as file:
                pickle.dump(d_hist, file)
                pickle.dump(g_hist, file)
            with open('losses.pickle', 'rb') as file:
                try:
                    dloss = []
                    gloss = []
                    while True:
                        dloss += pickle.load(file)
                        gloss += pickle.load(file)
                except EOFError:
                    print('Loss file read\nGenerating Plot')
                except FileNotFoundError:
                    print('Loss file not found')
            Utils().plot_history(dloss, gloss)

        else:
            generated = sess.run(gan.sample_images(8, 8))

            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            filename = os.path.join(images_dir, 'matrix_image.jpg')
            with open(filename, 'wb') as f:
                print('Image writen to %s' % filename)
                f.write(generated)


if __name__ == '__main__':
    tf.app.run()
