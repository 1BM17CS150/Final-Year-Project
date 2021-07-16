import os
import numpy as np
import tensorflow as tf
from tkinter import *
from tkinter.ttk import *
from glob import glob
from gan import GAN
from helping_functions import *

data_dir = '../TFrecord_generator/TFrecord_files'
log_dirr = 'checkpoints'
images_dir = 'images'
complete_dir = 'complete'
masktype = 'bottom'
batch_size = 128
latest_ckpt = 100000
nb_channels = 1
do_complete = True
num_per_epoch = 300
maxitr = 1001

CROP_SIZE = 96


def generateMask(masktype, image_shape, image_size):
    # Create mask
    if masktype == 'right':
        print("Selected mask is Right")
        scale = 0.6
        mask = np.ones(image_shape)
        sz = image_size
        l = int(sz * scale)
        u = int(sz)
        mask[:, l:u, :] = 0.0
    if masktype == 'left':
        print("Selected mask is Left")
        scale = 0.4
        mask = np.ones(image_shape)
        sz = image_size
        l = int(sz * scale)
        u = int(sz)
        mask[:, 0:l, :] = 0.0
    if masktype == 'bottom':
        print("Selected mask is Bottom")
        scale = 0.6
        mask = np.ones(image_shape)
        sz = image_size
        l = int(sz * scale)
        u = int(sz)
        mask[l:u, 0:u, :] = 0.0
    if masktype == 'top':
        print("Selected mask is Top")
        scale = 0.4
        mask = np.ones(image_shape)
        sz = image_size
        l = int(sz * scale)
        u = int(sz)
        mask[0:l, 0:u, :] = 0.0
    if masktype == 'diagonal':
        print("Selected mask is Diagonal")
        scale = 0.4
        mask = np.ones(image_shape)
        sz = image_size
        l = int(sz * scale)
        u = int(sz)
        for i in range(0, u):
            for j in range(0, u):
                if i > j:
                    mask[i, j, :] = 0.0
    if masktype == 'center':
        print("Selected mask is Center")
        scale = 0.25
        mask = np.ones(image_shape)
        sz = image_size
        l = int(sz * scale)
        u = int(sz * (1.0 - scale))
        mask[l:u, l:u, :] = 0.0
    if masktype == 'random':
        print("Selected mask is Random")
        fraction_masked = 0.8
        mask = np.ones(image_shape)
        mask[np.random.random(image_shape[:2]) < fraction_masked] = 0.0
    return mask


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


def main(root, progress, masktype, inputfile):
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

    g_saver = tf.train.Saver(gan.g.variables, max_to_keep=15)
    d_saver = tf.train.Saver(gan.d.variables, max_to_keep=15)
    g_checkpoint_restore_path = os.path.join(log_dirr, 'g.ckpt-' + str(latest_ckpt))
    d_checkpoint_restore_path = os.path.join(log_dirr, 'd.ckpt-' + str(latest_ckpt))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # restore or initialize generator
        if os.path.exists(g_checkpoint_restore_path + '.meta'):
            print('Restoring variables:')
            for v in gan.g.variables:
                print(' ' + v.name)
            g_saver.restore(sess, g_checkpoint_restore_path)
        else:
            print("Generator initialisation failed ")

        if do_complete:
            print("Completion Initiated")
            # restore discriminator
            if os.path.exists(d_checkpoint_restore_path + '.meta'):
                print('Restoring variables:')
                for v in gan.d.variables:
                    print(' ' + v.name)
                d_saver.restore(sess, d_checkpoint_restore_path)

                # Directory to save completed images
                if not os.path.exists(complete_dir):
                    print("complete folder created")
                    os.makedirs(complete_dir)
                else:
                    print("Complete directory exists")

                mask = generateMask(masktype, gan.image_shape, gan.image_size)

                # Read actual images
                originals = glob(inputfile)
                batch_mask = np.expand_dims(mask, axis=0)

                for idx in range(len(originals)):

                    print("Processing image %d" % (idx))
                    image_src = get_image(originals[idx], gan.image_size, nb_channels=nb_channels)
                    if nb_channels == 1:
                        image = np.expand_dims(np.expand_dims(image_src, axis=2), axis=0)

                    # Save original image (y)
                    filename = os.path.join(complete_dir, 'original_image_{:02d}.jpg'.format(idx))
                    imsave(image_src, filename)

                    # Save corrupted image (y . M)
                    filename = os.path.join(complete_dir, 'corrupted_image_{:02d}.jpg'.format(idx))
                    if nb_channels == 1:
                        masked_image = np.multiply(np.expand_dims(image_src, axis=2), mask)
                        imsave(masked_image[:, :, 0], filename)

                    zhat = np.random.uniform(-1, 1, size=(1, gan.z_dim))
                    v = 0
                    momentum = 0.9
                    lr = 0.001
                    for i in range(1, maxitr):
                        if i % 100 == 0:
                            print("Iteration %d" % (i))
                        try:
                            progress['value'] = i / (maxitr / 100)
                            root.update_idletasks()

                        except Exception as e:
                            print('Exited', e)
                            exit(0)
                        fd = {gan.zhat: zhat, gan.mask: batch_mask, gan.image: image}
                        run = [gan.complete_loss, gan.grad_complete_loss, gan.G]
                        _, g, G_imgs = sess.run(run, feed_dict=fd)

                        v_prev = np.copy(v)
                        v = momentum * v - lr * g[0]
                        zhat += -momentum * v_prev + (1 + momentum) * v
                        zhat = np.clip(zhat, -1, 1)

                        if i % 100 == 0:
                            filename = os.path.join(complete_dir,
                                                    'generated_img_{:02d}_{:04d}.jpg'.format(idx, i))
                            if nb_channels == 1:
                                save_images(G_imgs[0, :, :, 0], filename)

                            inv_masked_hat_image = np.multiply(G_imgs, 1.0 - batch_mask)
                            completed = masked_image + inv_masked_hat_image
                            filename = os.path.join(complete_dir,
                                                    'completed_{:02d}_{:04d}.jpg'.format(idx, i))
                            if nb_channels == 1:
                                save_images(completed[0, :, :, 0], filename)
            else:
                print("Discriminator initialisation failed. ")

        else:
            generated = sess.run(gan.sample_images(8, 8))

            if not os.path.exists(images_dir):
                os.makedirs(images_dir)

            filename = os.path.join(images_dir, 'generated_image.jpg')
            with open(filename, 'wb') as f:
                print('Image writen to %s' % filename)
                f.write(generated)

        print('Image Completion is done')


if __name__ == '__main__':
    tf.app.run()
