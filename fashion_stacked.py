import tensorflow as tf
import os
import datetime
import multiprocessing
import time

from fashion_helper import writer, name_in_checkpoint
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from nets import resnet_mine
import numpy as np
import cv2
from fashion_generator import DataLoader


class FashionAI(object):
    """
    FashionAI model
    2-stacked FPN version
    """

    def __init__(self):
        # initialization
        self.starter_learning_rate = 1e-4
        self.kernel_init = tf.variance_scaling_initializer()
        self.kernel_regularizer = None

        self.output_types = (tf.string, tf.float32, tf.float32, tf.float32)
        self.output_shapes = (tf.TensorShape([None]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None, 512, 512, 3]),
                              tf.TensorShape([None, 512, 512, 24]))
        # Build graph
        self._build_model()

    def _build_input(self):
        self.it = tf.data.Iterator.from_structure(self.output_types,
                                                  self.output_shapes)
        self.img_path, self.num_visible, self.img, self.heatmap = self.it.get_next()
        visible = tf.reduce_sum(self.heatmap, axis=1, keep_dims=True)
        visible = tf.reduce_sum(visible, axis=2, keep_dims=True)
        self.visible = tf.to_float(visible, name='ToFloat')

    def _build_fpn(self, img, name='fpn'):
        """
        First stage of model
        :param img: img from generator
        :param name: scope
        :return: logits
        """

        with tf.variable_scope(name_or_scope=name):

            with slim.arg_scope(slim.nets.resnet_v1.resnet_arg_scope()):
                resnet_out, end_points = nets.resnet_v1.resnet_v1_50(img, is_training=True)
            C5 = end_points[name + '/' + "resnet_v1_50/block4"]
            C4 = end_points[name + '/' + "resnet_v1_50/block3"]
            C3 = end_points[name + '/' + "resnet_v1_50/block2"]
            C2 = end_points[name + '/' + "resnet_v1_50/block1"]

            # print("Network Structure:")
            # for i in end_points:
            #     print("{}, {}".format(i, end_points[i].shape))
            # print(resnet_out)
            # print("===================================================")
            self.fpn1_C2 = C2

            C5_conv = tf.layers.conv2d(C5, 256, (1, 1), padding='same', activation=None,
                                       kernel_initializer=self.kernel_init,
                                       kernel_regularizer=self.kernel_regularizer)
            C5_conv = tf.layers.batch_normalization(C5_conv)
            C5_conv = tf.nn.relu(C5_conv)
            C4_conv = tf.layers.conv2d(C4, 256, (1, 1), padding='same', activation=None,
                                       kernel_initializer=self.kernel_init,
                                       kernel_regularizer=self.kernel_regularizer)
            C4_conv = tf.layers.batch_normalization(C4_conv)
            C4_conv = tf.nn.relu(C4_conv)
            C3_conv = tf.layers.conv2d(C3, 256, (1, 1), padding='same', activation=None,
                                       kernel_initializer=self.kernel_init,
                                       kernel_regularizer=self.kernel_regularizer)
            C3_conv = tf.layers.batch_normalization(C3_conv)
            C3_conv = tf.nn.relu(C3_conv)
            C2_conv = tf.layers.conv2d(C2, 256, (1, 1), padding='same', activation=None,
                                       kernel_initializer=self.kernel_init,
                                       kernel_regularizer=self.kernel_regularizer)
            C2_conv = tf.layers.batch_normalization(C2_conv)
            C2_conv = tf.nn.relu(C2_conv)

            upsample = tf.image.resize_bilinear(C5_conv, (16, 16))
            upsample = tf.layers.conv2d(upsample, 256, (1, 1),
                                        kernel_initializer=self.kernel_init,
                                        kernel_regularizer=self.kernel_regularizer)
            C4_conv += upsample

            upsample = tf.image.resize_bilinear(C4_conv, (32, 32))
            upsample = tf.layers.conv2d(upsample, 256, (1, 1),
                                        kernel_initializer=self.kernel_init,
                                        kernel_regularizer=self.kernel_regularizer)
            C3_conv += upsample

            upsample = tf.image.resize_bilinear(C3_conv, (64, 64))
            upsample = tf.layers.conv2d(upsample, 256, (1, 1),
                                        kernel_initializer=self.kernel_init,
                                        kernel_regularizer=self.kernel_regularizer)
            C2_conv += upsample

            C2_conv = tf.layers.conv2d(C2_conv, 256, [1, 1], activation=tf.nn.relu,
                                       kernel_initializer=self.kernel_init,
                                       kernel_regularizer=self.kernel_regularizer)
            logits = tf.layers.conv2d(C2_conv, 24, [3, 3], padding='same',
                                      kernel_initializer=self.kernel_init,
                                      kernel_regularizer=self.kernel_regularizer)
            self.fpn1_C2_conv = C2_conv

            logits = tf.image.resize_bilinear(logits, (512, 512))

        return logits

    def _build_fpn2(self, inputs, name='fpn2'):
        """
        :param inputs: C2_conv from fpn1
        :param name: scope
        :return: logits
        """

        with tf.variable_scope(name_or_scope=name):

            # slightly modified from slim implementation of ResNet
            with slim.arg_scope(slim.nets.resnet_v1.resnet_arg_scope()):
                resnet_out, end_points = resnet_mine.resnet_v1_50(inputs, is_training=True)
            C5 = end_points[name + '/' + "resnet_v1_50/block4"]
            C4 = end_points[name + '/' + "resnet_v1_50/block3"]
            C3 = end_points[name + '/' + "resnet_v1_50/block2"]

            C5_conv = tf.layers.conv2d(C5, 256, (1, 1), padding='same', activation=None,
                                       kernel_initializer=self.kernel_init,
                                       kernel_regularizer=self.kernel_regularizer)
            C5_conv = tf.layers.batch_normalization(C5_conv)
            C5_conv = tf.nn.relu(C5_conv)
            C4_conv = tf.layers.conv2d(C4, 256, (1, 1), padding='same', activation=None,
                                       kernel_initializer=self.kernel_init,
                                       kernel_regularizer=self.kernel_regularizer)
            C4_conv = tf.layers.batch_normalization(C4_conv)
            C4_conv = tf.nn.relu(C4_conv)
            C3_conv = tf.layers.conv2d(C3, 256, (1, 1), padding='same', activation=None,
                                       kernel_initializer=self.kernel_init,
                                       kernel_regularizer=self.kernel_regularizer)
            C3_conv = tf.layers.batch_normalization(C3_conv)
            C3_conv = tf.nn.relu(C3_conv)

            upsample = tf.image.resize_bilinear(C5_conv, (16, 16))
            upsample = tf.layers.conv2d(upsample, 256, (1, 1),
                                        kernel_initializer=self.kernel_init,
                                        kernel_regularizer=self.kernel_regularizer)
            C4_conv += upsample

            upsample = tf.image.resize_bilinear(C4_conv, (32, 32))
            upsample = tf.layers.conv2d(upsample, 256, (1, 1),
                                        kernel_initializer=self.kernel_init,
                                        kernel_regularizer=self.kernel_regularizer)
            C3_conv += upsample

            upsample = tf.image.resize_bilinear(C3_conv, (64, 64))
            upsample = tf.layers.conv2d(upsample, 256, (1, 1),
                                        kernel_initializer=self.kernel_init,
                                        kernel_regularizer=self.kernel_regularizer)

            C2_conv = inputs + upsample

            C2_conv = tf.layers.conv2d(C2_conv, 256, [1, 1], activation=tf.nn.relu,
                                       kernel_initializer=self.kernel_init,
                                       kernel_regularizer=self.kernel_regularizer)
            logits = tf.layers.conv2d(C2_conv, 24, [3, 3], padding='same',
                                      kernel_initializer=self.kernel_init,
                                      kernel_regularizer=self.kernel_regularizer)

            logits = tf.image.resize_bilinear(logits, (512, 512))

        return logits

    def _build_solver(self):
        """
        Build Optimizer
        """
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.starter_learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                   20000, 0.9, staircase=True)
        solver = tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            solver.minimize(self.loss, global_step=self.global_step)

    def _build_model(self):
        self._build_input()

        logits1 = self._build_fpn(self.img, name='fpn1')
        self.pred_heatmap1 = tf.nn.sigmoid(logits1)
        self.pred_heatmap1 *= self.visible

        with tf.variable_scope(name_or_scope='inputs_fpn2'):
            inputs_fpn2 = tf.layers.conv2d(self.fpn1_C2_conv, 256, [1, 1], activation=tf.nn.relu,
                                           kernel_initializer=self.kernel_init,
                                           kernel_regularizer=self.kernel_regularizer)
            inputs_fpn2 += tf.layers.conv2d(self.fpn1_C2, 256, [1, 1], activation=tf.nn.relu,
                                            kernel_initializer=self.kernel_init,
                                            kernel_regularizer=self.kernel_regularizer)
        logits2 = self._build_fpn2(inputs_fpn2, name='fpn2')
        self.pred_heatmap2 = tf.nn.sigmoid(logits2)
        self.pred_heatmap2 *= self.visible

        with tf.variable_scope('loss'):
            self.loss1 = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.heatmap,
                                                         logits=logits1,
                                                         weights=self.visible)
            self.loss2 = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.heatmap,
                                                         logits=logits2,
                                                         weights=self.visible)
            self.loss = self.loss1 + self.loss2
        self._build_solver()
        self._build_summary()

    def _build_summary(self):
        """
        Summary for visualization using tensorboard.
        """
        tf.summary.image("image", self.img)
        tf.summary.scalar("loss_total", self.loss)
        tf.summary.scalar("loss1", self.loss1)
        tf.summary.scalar("loss2", self.loss2)

        tf.summary.image("heatmap_ground_truth",
                         tf.reduce_sum(self.heatmap, axis=3, keep_dims=True))
        tf.summary.image("heatmap_predicted_1",
                         tf.reduce_sum(self.pred_heatmap1, axis=3, keep_dims=True))
        tf.summary.image("heatmap_predicted_2",
                         tf.reduce_sum(self.pred_heatmap2, axis=3, keep_dims=True))
        self.merged = tf.summary.merge_all()

    def train(self,
              max_epochs=20,
              model_dir=None,
              dataset_dir='train_set',
              batch_size=10,
              write_summary=False,
              freq_summary=200):
        """
        train
        :param max_epochs: int
        :param model_dir: str, continue training from pre-trained checkpoints,
                          if None, training from ImageNet checkpoints
        :param dataset_dir: str
        :param batch_size: int
        :param write_summary: boolean
        :param freq_summary: int
        """

        loader = DataLoader(data_dir=dataset_dir, mode='train')
        dataset = tf.data.Dataset.from_generator(generator=loader.generator,
                                                 output_types=(tf.string,
                                                               tf.float32,
                                                               tf.float32,
                                                               tf.float32),
                                                 output_shapes=(tf.TensorShape([]),
                                                                tf.TensorShape([]),
                                                                tf.TensorShape([512, 512, 3]),
                                                                tf.TensorShape([512, 512, 24])))
        dataset = dataset.shuffle(100)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(20)

        train_init = self.it.make_initializer(dataset)

        print("training starts.")
        variables_to_restore = slim.get_model_variables()
        # for var in variables_to_restore:
        #     print(var.op.name)
        variables_to_restore = {name_in_checkpoint(var): var for var in variables_to_restore}

        restorer = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            now = datetime.datetime.now()
            save_dir = 'model/fashion_stacked_{}_{}/fashionai.ckpt'.format(now.month, now.day)
            summary_dir = 'summary/model_stacked_{}_{}_{}:{}'.format(now.month, now.day, now.hour, now.minute)
            train_writer = tf.summary.FileWriter(summary_dir, sess.graph)
            # continue training
            if model_dir:
                print("continue training from " + model_dir)
                saver.restore(sess, model_dir)
            else:
                sess.run(tf.global_variables_initializer())
                restorer.restore(sess, 'model/resnet_v1_50.ckpt')
            # train
            for epoch in range(max_epochs):
                sess.run(train_init)

                print("epoch {} begins:".format(epoch))
                try:
                    while True:
                        if write_summary:
                            _, loss, summary, step = sess.run([self.train_op,
                                                               self.loss,
                                                               self.merged,
                                                               self.global_step])
                            if step % freq_summary == 0:
                                # summary
                                train_writer.add_summary(summary, step)
                        else:
                            _, loss, step = sess.run([self.train_op, self.loss, self.global_step])
                        # print("Epoch {} step {}: {}".format(epoch, step, loss))
                        if step % 500 == 0:
                            print('saving checkpoint......')
                            saver.save(sess, save_dir)
                            print('checkpoint saved.')
                except tf.errors.OutOfRangeError:
                    print('saving checkpoint......')
                    saver.save(sess, save_dir)
                    print('checkpoint saved.')

    def test(self,
             data_dir='test_b',
             model_dir=None,
             output_dir=None):
        """
        test
        :param data_dir: str
        :param model_dir: str, trained checkpoint
        :param output_dir: str, predict files
        """
        print("testing starts.")
        test_loader = DataLoader(data_dir=data_dir, mode='test')
        testset = tf.data.Dataset.from_generator(generator=test_loader.generator,
                                                 output_types=(tf.string,
                                                               tf.float32,
                                                               tf.float32,
                                                               tf.float32),
                                                 output_shapes=(tf.TensorShape([]),
                                                                tf.TensorShape([]),
                                                                tf.TensorShape([512, 512, 3]),
                                                                tf.TensorShape([512, 512, 24])))
        testset = testset.batch(2)
        testset = testset.prefetch(10)
        test_init = self.it.make_initializer(testset)

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, model_dir)
            sess.run(test_init)
            queue = multiprocessing.Queue(maxsize=30)
            writer_process = multiprocessing.Process(target=writer, args=[output_dir, queue, 'stop'])
            writer_process.start()
            print('writing predictions...')
            try:
                while True:
                    img_path, heatmaps = sess.run([self.img_path, self.pred_heatmap2])
                    queue.put(('continue', img_path, heatmaps))

            except tf.errors.OutOfRangeError:
                queue.put(('stop', None, None))

        writer_process.join()
        print('testing finished.')

# Basic use
# if __name__ == '__main__':
#     fashionAI = FashionAI()
#     fashionAI.train(max_epochs=20, batch_size=10,
#                     write_summary=True, freq_summary=10,
#                     model_dir='model/ai0515/fashionai.ckpt',
#                     )
#     fashionAI.test(model_dir='model/fashion_stacked/fashionai.ckpt', output_dir="outputs/results_test_bb.csv")
