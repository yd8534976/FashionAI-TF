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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class FashionAI(object):

    def __init__(self,
                 is_training=False):
        self.starter_learning_rate = 1e-4
        self.kernel_init = tf.variance_scaling_initializer()
        # self.kernel_init = tf.truncated_normal_initializer(stddev=0.02)
        self.kernel_regularizer = None
        self.is_training = is_training

        self.output_types = (tf.string, tf.float32, tf.float32)
        self.output_shapes = (tf.TensorShape([None]),
                              tf.TensorShape([None, 512, 512, 3]),
                              tf.TensorShape([None, 512, 512, 24]))
        self._build_model()

    def _build_input(self):
        self.it = tf.data.Iterator.from_structure(self.output_types,
                                                  self.output_shapes)
        self.img_path, self.img, self.heatmap = self.it.get_next()
        visible = tf.reduce_max(self.heatmap, axis=1, keep_dims=True)
        self.visible = tf.reduce_max(visible, axis=2, keep_dims=True)

    def _conv_bn_relu(self, x, name="conv_bn_relu"):
        with tf.variable_scope(name_or_scope=name):
            x = tf.layers.conv2d(x, 256, (1, 1),
                                 kernel_initializer=self.kernel_init,
                                 kernel_regularizer=self.kernel_regularizer)
            x = tf.layers.batch_normalization(x, training=self.is_training)
            x = tf.nn.relu(x)
        return x

    def _connect_block(self, lateral, up, name="connect_block"):
        with tf.variable_scope(name_or_scope=name):
            shape = tf.shape(lateral)
            up = tf.image.resize_bilinear(up, shape[1:3])
            up = tf.layers.conv2d(up, 256, (1, 1),
                                  kernel_initializer=self.kernel_init,
                                  kernel_regularizer=self.kernel_regularizer)
            out = lateral + up
        return out

    def _build_fpn(self, img, name='fpn'):

        with tf.variable_scope(name_or_scope=name):

            with slim.arg_scope(slim.nets.resnet_v1.resnet_arg_scope()):
                resnet_out, end_points = nets.resnet_v1.resnet_v1_50(img, is_training=self.is_training)
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

            C5_conv = self._conv_bn_relu(C5, name="conv_bn_relu_1")
            C4_conv = self._conv_bn_relu(C4, name="conv_bn_relu_2")
            C3_conv = self._conv_bn_relu(C3, name="conv_bn_relu_3")
            C2_conv = self._conv_bn_relu(C2, name="conv_bn_relu_4")

            out = self._connect_block(lateral=C4_conv,
                                      up=C5_conv,
                                      name="connect_block1")

            out = self._connect_block(lateral=C3_conv,
                                      up=out,
                                      name="connect_block2")

            out = self._connect_block(lateral=C2_conv,
                                      up=out,
                                      name="connect_block3")

            features = tf.layers.conv2d(out, 256, [1, 1], activation=tf.nn.relu,
                                        kernel_initializer=self.kernel_init,
                                        kernel_regularizer=self.kernel_regularizer)
            logits = tf.layers.conv2d(features, 24, [3, 3], padding='same',
                                      kernel_initializer=self.kernel_init,
                                      kernel_regularizer=self.kernel_regularizer)
            logits = tf.image.resize_bilinear(logits, (512, 512))

        return logits, features

    def _build_fpn2(self, inputs, name='fpn2'):
        """
        :param inputs: features from fpn1
        :param name: scope
        :return: predicted heatmap
        """

        with tf.variable_scope(name_or_scope=name):

            with slim.arg_scope(slim.nets.resnet_v1.resnet_arg_scope()):
                resnet_out, end_points = resnet_mine.resnet_v1_50(inputs, is_training=self.is_training)
            C5 = end_points[name + '/' + "resnet_v1_50/block4"]
            C4 = end_points[name + '/' + "resnet_v1_50/block3"]
            C3 = end_points[name + '/' + "resnet_v1_50/block2"]

            C5_conv = self._conv_bn_relu(C5, name="conv_bn_relu_1")
            C4_conv = self._conv_bn_relu(C4, name="conv_bn_relu_2")
            C3_conv = self._conv_bn_relu(C3, name="conv_bn_relu_3")

            out = self._connect_block(lateral=C4_conv,
                                      up=C5_conv,
                                      name="connect_block1")

            out = self._connect_block(lateral=C3_conv,
                                      up=out,
                                      name="connect_block2")

            out = self._connect_block(lateral=inputs,
                                      up=out,
                                      name="connect_block3")

            features = tf.layers.conv2d(out, 256, [1, 1], activation=tf.nn.relu,
                                        kernel_initializer=self.kernel_init,
                                        kernel_regularizer=self.kernel_regularizer)
            logits = tf.layers.conv2d(features, 24, [3, 3], padding='same',
                                      kernel_initializer=self.kernel_init,
                                      kernel_regularizer=self.kernel_regularizer)

            logits = tf.image.resize_bilinear(logits, (512, 512))

        return logits, features

    def _build_solver(self):
        self.global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = self.starter_learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                   2000, 0.9, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss,
                                                                           global_step=self.global_step)

    def _build_model(self):

        self._build_input()

        logits1, features1 = self._build_fpn(self.img, name='fpn1')
        self.pred_heatmap1 = tf.nn.sigmoid(logits1)
        self.pred_heatmap1 *= self.visible

        with tf.variable_scope(name_or_scope='inputs_fpn2'):
            inputs_fpn2 = tf.layers.conv2d(features1, 256, [1, 1], activation=tf.nn.relu,
                                           kernel_initializer=self.kernel_init,
                                           kernel_regularizer=self.kernel_regularizer)
            inputs_fpn2 += tf.layers.conv2d(self.fpn1_C2, 256, [1, 1], activation=tf.nn.relu,
                                            kernel_initializer=self.kernel_init,
                                            kernel_regularizer=self.kernel_regularizer)

        logits2, features2 = self._build_fpn2(inputs_fpn2, name='fpn2')
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

    def train(self, max_epochs=20, model_dir=None, dataset_dir='train_set',
              batch_size=4,
              write_summary=False, freq_summary=200):

        loader = DataLoader(data_dir=dataset_dir, mode='train')
        dataset = tf.data.Dataset.from_generator(generator=loader.generator,
                                                 output_types=(tf.string,
                                                               tf.float32,
                                                               tf.float32),
                                                 output_shapes=(tf.TensorShape([]),
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
        # config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            now = datetime.datetime.now()
            save_dir = 'model/{}_{}/'.format(now.month, now.day)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            save_dir += 'fashion_stacked/fashionai.ckpt'

            summary_dir = 'summary/{}_{}/model_stacked_{}:{}'.format(now.month, now.day, now.hour, now.minute)
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
                            # evaluator.evaluate(pred_heatmap, label, img_path)
                            if step % freq_summary == 0:
                                # summary
                                train_writer.add_summary(summary, step)
                        else:
                            _, loss, step = sess.run([self.train_op, self.loss, self.global_step])
                        # print("Epoch {} step {}: {}".format(epoch, step, loss))
                        if step % 1000 == 0:
                            print('saving checkpoint......')
                            saver.save(sess, save_dir)
                            print('checkpoint saved.')
                except tf.errors.OutOfRangeError:
                    print('saving checkpoint......')
                    saver.save(sess, save_dir)
                    print('checkpoint saved.')

    def test(self, data_dir='test_b', model_dir=None, output_dir=None):
        print("testing starts.")
        test_loader = DataLoader(data_dir=data_dir, mode='test')
        testset = tf.data.Dataset.from_generator(generator=test_loader.generator,
                                                 output_types=(tf.string,
                                                               tf.float32,
                                                               tf.float32),
                                                 output_shapes=(tf.TensorShape([]),
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


# if __name__ == '__main__':
#     fashionAI = FashionAI(is_training=True)
#     fashionAI.train(max_epochs=20, batch_size=5,
#                     write_summary=True, freq_summary=10,
#                     model_dir=None,
#                     )
#     # fashionAI.test(model_dir='model/fashion_stacked/fashionai.ckpt', output_dir="outputs/results_test_bb.csv")
