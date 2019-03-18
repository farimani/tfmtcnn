# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
import cv2
import tensorflow as tf
# from tensorflow.contrib import slim
# from datetime import datetime

import mef
import mef_tf


from tfmtcnn.trainers.AbstractNetworkTrainer import AbstractNetworkTrainer
from tfmtcnn.trainers.ModelEvaluator import ModelEvaluator

from tfmtcnn.losses.class_loss_ohem import class_loss_ohem
from tfmtcnn.losses.bounding_box_loss_ohem import bounding_box_loss_ohem
from tfmtcnn.losses.landmark_loss_ohem import landmark_loss_ohem

from tfmtcnn.datasets.TensorFlowDataset import TensorFlowDataset
# import tfmtcnn.datasets.constants as datasets_constants

from tfmtcnn.networks.NetworkFactory import NetworkFactory
# from tfmtcnn.networks.FaceDetector import FaceDetector


class SimpleNetworkTrainer(AbstractNetworkTrainer):
    def __init__(self, network_name='PNet', batch_size=-1):
        AbstractNetworkTrainer.__init__(self, network_name, batch_size=batch_size)
        self._session = None
        self._model_evaluator = ModelEvaluator()

    def _train_model(self, base_learning_rate, loss):
        self._global_step = tf.Variable(0, name='global_step', trainable=False)

        number_of_iterations = int(self._number_of_samples / self._batch_size)
        learning_rate_op = tf.train.exponential_decay(
            base_learning_rate,
            self._global_step, (number_of_iterations * 2),
            0.94,
            staircase=True)

        # optimizer = tf.train.MomentumOptimizer(learning_rate_op, 0.9)
        # optimizer = tf.train.optimizer = tf.train.RMSPropOptimizer(learning_rate_op, decay=0.9,
        #                                                            momentum=0.9, epsilon=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate_op,
                                           # base_learning_rate,
                                           beta1=0.9, beta2=0.999, epsilon=1e-08)
        train_op = optimizer.minimize(loss, global_step=self._global_step)
        return train_op, learning_rate_op

    @staticmethod
    def _random_flip_images(image_batch, label_batch, landmark_batch):
        if random.choice([0, 1]) > 0:
            # num_images = image_batch.shape[0]
            fliplandmarkindexes = np.where(label_batch == -2)[0]
            flipposindexes = np.where(label_batch == 1)[0]
            flipindexes = np.concatenate((fliplandmarkindexes, flipposindexes))

            for i in flipindexes:
                cv2.flip(image_batch[i], 1, image_batch[i])

            for i in fliplandmarkindexes:
                landmark_ = landmark_batch[i].reshape((-1, 2))
                landmark_ = np.asarray([(1 - x, y) for (x, y) in landmark_])
                landmark_[[0, 1]] = landmark_[[1, 0]]
                landmark_[[3, 4]] = landmark_[[4, 3]]
                landmark_batch[i] = landmark_.ravel()

        return image_batch, landmark_batch

    @staticmethod
    def _calculate_accuracy(cls_prob, label):
        pred = tf.argmax(cls_prob, axis=1)
        label_int = tf.cast(label, tf.int64)
        cond = tf.where(tf.greater_equal(label_int, 0))
        picked = tf.squeeze(cond)
        label_picked = tf.gather(label_int, picked)
        pred_picked = tf.gather(pred, picked)
        accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked, pred_picked), tf.float32))
        return accuracy_op

    def _read_data(self, dataset_root_dir):
        dataset_dir = self.dataset_dir(dataset_root_dir)
        tensorflow_file_name = self._image_list_file_name(dataset_dir)

        mef.tsprint(f"Obtaining number of samples from TFRecords file {tensorflow_file_name}...")
        self._number_of_samples = mef_tf.get_num_tfrecords(tensorflow_file_name, show_progress=1000)
        mef.tsprint(f"{self._number_of_samples} records in TFRecords file.")

        image_size = self.network_size()
        tensorflow_dataset = TensorFlowDataset()
        return tensorflow_dataset.read_tensorflow_file(tensorflow_file_name, self._batch_size, image_size)

    def _evaluate(self, network_name, model_root_dir):
        if not self._model_evaluator.create_detector(network_name, model_root_dir):
            return False

        return self._model_evaluator.evaluate(print_result=True)

    def load_test_dataset(self, dataset_name, annotation_image_dir, annotation_file_name):
        return self._model_evaluator.load(dataset_name, annotation_image_dir, annotation_file_name)

    def train(self, network_name, dataset_root_dir, train_root_dir,
              base_learning_rate, max_number_of_epoch, log_every_n_steps):
        network_train_dir = self.network_train_dir(train_root_dir)
        mef.create_dir_if_necessary(network_train_dir, raise_on_error=True)

        image_size = self.network_size()
        image_batch, label_batch, bbox_batch, landmark_batch = self._read_data(dataset_root_dir)
        class_loss_ratio, bbox_loss_ratio, landmark_loss_ratio = NetworkFactory.loss_ratio(network_name)

        input_image = tf.placeholder(tf.float32, shape=[self._batch_size, image_size, image_size, 3],
                                     name='input_image')
        target_label = tf.placeholder(tf.float32, shape=[self._batch_size], name='target_label')
        target_bounding_box = tf.placeholder(tf.float32, shape=[self._batch_size, 4], name='target_bounding_box')
        target_landmarks = tf.placeholder(tf.float32, shape=[self._batch_size, 10], name='target_landmarks')

        output_class_probability, output_bounding_box, \
            output_landmarks = self._network.setup_training_network(input_image)

        class_loss_op = class_loss_ohem(output_class_probability, target_label)
        bounding_box_loss_op = bounding_box_loss_ohem(output_bounding_box, target_bounding_box, target_label)
        landmark_loss_op = landmark_loss_ohem(output_landmarks, target_landmarks, target_label)

        class_accuracy_op = self._calculate_accuracy(output_class_probability, target_label)
        l2_loss_op = tf.add_n(tf.losses.get_regularization_losses())

        total_loss = class_loss_ratio * class_loss_op + bbox_loss_ratio * bounding_box_loss_op + \
            landmark_loss_ratio * landmark_loss_op + l2_loss_op
        train_op, learning_rate_op = self._train_model(base_learning_rate, total_loss)

        # self._session = tf.Session()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        cfg = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options, log_device_placement=False)
        self._session = tf.Session(config=cfg)

        # saver = tf.train.Saver(save_relative_paths=True, max_to_keep=3)
        saver = tf.train.Saver(save_relative_paths=True, max_to_keep=0)  # All checkpoint files are kept.
        self._session.run(tf.global_variables_initializer())

        tf.summary.scalar("class_loss", class_loss_op)
        tf.summary.scalar("bounding_box_loss", bounding_box_loss_op)
        tf.summary.scalar("landmark_loss", landmark_loss_op)
        tf.summary.scalar("class_accuracy", class_accuracy_op)
        summary_op = tf.summary.merge_all()

        logs_dir = os.path.join(network_train_dir, "logs")
        mef.create_dir_if_necessary(logs_dir, raise_on_error=True)

        summary_writer = tf.summary.FileWriter(logs_dir, self._session.graph)
        coordinator = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=self._session, coord=coordinator)
        max_number_of_steps = int(self._number_of_samples / self._batch_size + 1) * max_number_of_epoch
        global_step = current_step = epoch = 0
        skip_model_saving = False
        if self._network.load_model(self._session, network_train_dir):
            model_path = self._network.model_path()
            global_step = tf.train.global_step(self._session, self._global_step)
            epoch = int(global_step * self._batch_size / self._number_of_samples)
            # skip_model_saving = True        # MEF: Why?!
            mef.tsprint(f"Model is restored from model path - {model_path} with global step - {global_step}.")

        network_train_file_name = os.path.join(network_train_dir, self.network_name())
        self._session.graph.finalize()

        mef.tsprint(f"Starting training: {max_number_of_steps - global_step} steps left. "
                    f"Batch size: {self.batch_size()}, total samples: {self._number_of_samples}, "
                    f"num epochs: {max_number_of_epoch}.")
        pt = mef.ProgressText(max_number_of_steps, current=global_step)

        try:
            for step in range(global_step, max_number_of_steps):
                if step == max_number_of_steps - 2:
                    print("jere")

                current_step += 1
                if coordinator.should_stop():
                    break

                image_batch_array, label_batch_array, \
                    bbox_batch_array, landmark_batch_array = self._session.run([image_batch, label_batch, bbox_batch,
                                                                                landmark_batch])
                image_batch_array, landmark_batch_array = self._random_flip_images(image_batch_array,
                                                                                   label_batch_array,
                                                                                   landmark_batch_array)
                _, _, summary = self._session.run([train_op, learning_rate_op, summary_op],
                                                  feed_dict={
                                                      input_image: image_batch_array,
                                                      target_label: label_batch_array,
                                                      target_bounding_box: bbox_batch_array,
                                                      target_landmarks: landmark_batch_array
                                                  })
                pt.update(f"{step+1}/{max_number_of_steps}")

                if (step + 1) % log_every_n_steps == 0:
                    tensor_list = [class_loss_op, bounding_box_loss_op, landmark_loss_op, l2_loss_op,
                                   learning_rate_op, class_accuracy_op]
                    feed_dict = {
                            input_image: image_batch_array,
                            target_label: label_batch_array,
                            target_bounding_box: bbox_batch_array,
                            target_landmarks: landmark_batch_array
                        }
                    current_class_loss, current_bbox_loss, current_landmark_loss, \
                        current_l2_loss, current_lr, current_accuracy = self._session.run(tensor_list,
                                                                                          feed_dict=feed_dict)
                    mef.tsprint(f"\r(epoch - {epoch+1}, step - {step+1}) - (accuracy - {current_accuracy:3f}, "
                                f"class loss - {current_class_loss:.4f}, bbox loss - {current_bbox_loss:4f}, "
                                f"landmark loss - {current_landmark_loss:.4f}, L2 loss - {current_l2_loss:4f}, "
                                f"lr - {current_lr})")
                    summary_writer.add_summary(summary, global_step=global_step)

                if current_step * self._batch_size >= self._number_of_samples:
                    if skip_model_saving:
                        skip_model_saving = False
                    else:
                        mef.tsprint(f"\rEpoch: {epoch+1}, Step: {step+1}. "
                                    f"Saving checkpoint to {network_train_file_name}")
                        saver.save(self._session, network_train_file_name, global_step=self._global_step)
                        mef.tsprint(f"Model saved. Evaluating model...")
                        self._evaluate(network_name, train_root_dir)
                        mef.tsprint(f"Evaluation complete.")

                    epoch += 1
                    current_step = 0

        except tf.errors.OutOfRangeError:
            print("Error")
        finally:
            coordinator.request_stop()
            summary_writer.close()
        coordinator.join(threads)
        self._session.close()
        return True
