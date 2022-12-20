import os
import cv2
#import tensorflow.compat.v1 as tf
#import tensorflow.compat.v1.keras as keras
import tensorflow as tf
import datetime
import numpy as np
import math
import matplotlib.pyplot as plt
# import onnx
# import keras2onnx
import pandas as pd

from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model

class ResNet(object):
    def __init__(self):
        self.model_name ='ResNet'

    def lr_schedule(self, epoch):
        lr = 1e-3
        if epoch > 180:
            lr *= 0.5e-3
        elif epoch > 160:
            lr *= 1e-3
        elif epoch > 120:
            lr *= 1e-2
        elif epoch > 80:
            lr *= 1e-1
        print('Learning rate : ', lr)
        return lr

    def ResnetLayer(self,
                    inputs,
                    num_filters=16,
                    kernel_size=3,
                    strides=1,
                    activation='relu',
                    batch_normalization=True,
                    conv_first=True):
        conv = Conv2D(num_filters,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(1e-4))
        x = inputs
        if conv_first:
            x = conv(x)
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
        else:
            if batch_normalization:
                x = BatchNormalization()(x)
            if activation is not None:
                x = Activation(activation)(x)
            x = conv(x)
        return x

    def ResnetV1(self, input_shape, depth, num_classes=10):
        if (depth - 2) % 6 != 0:
            raise ValueError('depth should be 6n+2 (eg 20, 32, in [a])')

        num_filters = 16
        num_res_blocks = int((depth - 2) / 6)

        inputs = Input(shape=input_shape)
        x = self.ResnetLayer(inputs=inputs)

        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:
                    strides = 2
                y = self.ResnetLayer(inputs=x,
                                     num_filters=num_filters,
                                     strides=strides)
                y = self.ResnetLayer(inputs=y,
                                     num_filters=num_filters,
                                     activation=None)

                if stack > 0 and res_block == 0:
                    x = self.ResnetLayer(inputs=x,
                                         num_filters=num_filters,
                                         kernel_size=1,
                                         strides=strides,
                                         activation=None,
                                         batch_normalization=False)
                x = add([x, y])
                x = Activation('relu')(x)
            num_filters *= 2

        x = AveragePooling2D(pool_size=4)(x)
        y = Flatten()(x)
        y = Dense(units=32, activation='relu')(y)
        y = Dropout(0.5)(y)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def ResnetV2(self, input_shape, depth, num_classes=10):
        if (depth -2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 110 in [b]')
        num_filters_in = 16
        num_res_blocks = int((depth - 2) / 9)

        inputs = Input(shape=input_shape)

        x = self.ResnetLayer(inputs=inputs,
                        num_filters=num_filters_in,
                        conv_first=True)

        for stage in range(3):
            for res_block in range(num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block == 0:
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block == 0:
                        strides = 2

                print("first ResnetLayer %d" % strides)
                y = self.ResnetLayer(inputs=x,
                                num_filters=num_filters_in,
                                kernel_size=1,
                                strides=strides,
                                activation=activation,
                                batch_normalization=batch_normalization,
                                conv_first=False)
                y = self.ResnetLayer(inputs=y,
                                num_filters=num_filters_in,
                                conv_first=False)
                y = self.ResnetLayer(inputs=y,
                                num_filters=num_filters_out,
                                kernel_size=1,
                                conv_first=False)
                if res_block == 0:
                    print("4th ResnetLayer %d" % strides)
                    x = self.ResnetLayer(inputs=x,
                                    num_filters=num_filters_out,
                                    kernel_size=1,
                                    strides=strides,
                                    activation=activation,
                                    batch_normalization=False)
                x = add([x, y])

            num_filters_in = num_filters_out

        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = AveragePooling2D(pool_size=4)(x)
        y = Flatten()(x)
        y = Dense(units=128, activation='relu')(y)
        y = Dropout(0.5)(y)
        y = Dense(units=64, activation='relu')(y)
        y = Dropout(0.5)(y)
        outputs = Dense(num_classes,
                        activation='softmax',
                        kernel_initializer='he_normal')(y)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    """
    Training for ResNet version 1
    """
    def Train(self, version=1):
        batch_size = 100
        epochs = 100
        num_classes = 10

        n = 3

        if version == 1:
            depth = n * 6 + 2
        elif version == 2:
            depth = n * 9 + 2

        model_type = 'ResNet%dv%d' % (depth, version)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # img_array = []
        # v = GetFilePath()
        # for file in v:
        #     strings = file.split('\\')
        #     if strings[1] == "subject_4":
        #         break
        #     mnist_imgs = ResizeImage(file)
        #     img_array.append(mnist_imgs)
        #
        # res = MultiImageTesting('mnist_model_modified_epoch100', img_array)
        #
        # i = 0
        # img_array_tl = []
        # img_array_tl_res = []
        # for file in v:
        #     strings = file.split('\\')
        #     if strings[1] == "subject_4":
        #         break
        #     if int(strings[-2]) == int(res[i]):
        #         img_array_tl.append(img_array[i])
        #         img_array_tl_res.append(res[i])
        #     i = i + 1
        #
        # x_train = np.append(x_train, np.array(img_array_tl), axis=0)
        # y_train = np.append(y_train, np.array(img_array_tl_res), axis=0)

        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        y_train = y_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        y_test = y_test.reshape(-1, 1)

        input_shape = x_train.shape[1:]

        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        print('y_train shape:', y_train.shape)

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        if version == 2:
            model = self.ResnetV2(input_shape=input_shape, depth=depth)
        else:
            model = self.ResnetV1(input_shape=input_shape, depth=depth)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.lr_schedule(0)),
                      metrics=['acc'])

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.summary()

        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'mnist_%s_model.{epoch:03d}.h5' % model_type
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True)

        lr_scheduler = LearningRateScheduler(self.lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        callbacks = [checkpoint, lr_reducer, lr_scheduler, tensorboard_callback]

        steps_per_epoch = math.ceil(len(x_train) / batch_size)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  verbose=1,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks)
        scores = model.evaluate(x_test,
                                y_test,
                                batch_size=batch_size,
                                verbose=0)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    """
    Training for ResNet version 1 with transfer learning
    """
    def TrainingResNetTransferLearning(self, version=1, basemodel_name=''):
        batch_size = 100
        epochs = 200
        num_classes = 10

        n = 3

        if version == 1:
            depth = n * 6 + 2
        elif version == 2:
            depth = n * 9 + 2

        model_type = 'ResNet%dv%d' % (depth, version)

        model = tf.keras.models.load_model(basemodel_name)

        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        img_array = []
        v = self.GetFilePath()
        for file in v:
            strings = file.split('\\')
            if strings[1] == "subject_4":
                break
            mnist_imgs = self.ResizeImage(file)
            img_array.append(mnist_imgs)

        res = self.MultiImageTesting(basemodel_name, img_array)

        i = 0
        img_array_tl = []
        img_array_tl_res = []
        for file in v:
            strings = file.split('\\')
            if strings[1] == "subject_4":
                break
            if int(strings[-2]) == int(res[i]):
                img_array_tl.append(img_array[i])
                img_array_tl_res.append(res[i])
            i = i + 1

        x_train = np.array(img_array_tl)
        y_train = np.array(img_array_tl_res)

        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train.reshape(-1, 28, 28, 1)
        y_train = y_train.reshape(-1, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        y_test = y_test.reshape(-1, 1)

        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(lr=self.lr_schedule(0)),
                      metrics=['acc'])

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.summary()

        save_dir = os.path.join(os.getcwd(), 'saved_models')
        model_name = 'mnist_%s_model.{epoch:03d}_tl.h5' % model_type
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        checkpoint = ModelCheckpoint(filepath=filepath,
                                     monitor='val_acc',
                                     verbose=1,
                                     save_best_only=True)

        lr_scheduler = LearningRateScheduler(self.lr_schedule)

        lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                                       cooldown=0,
                                       patience=5,
                                       min_lr=0.5e-6)
        callbacks = [checkpoint, lr_reducer, lr_scheduler, tensorboard_callback]

        steps_per_epoch = math.ceil(len(x_train) / batch_size)
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  verbose=1,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  steps_per_epoch=steps_per_epoch,
                  callbacks=callbacks)
        scores = model.evaluate(x_test,
                                y_test,
                                batch_size=batch_size,
                                verbose=0)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

    def ResizeImage(self, filename):
        img = cv2.imread(filename)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        ret, img_th = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)

        resized_img = cv2.resize(img_th, dsize=(28, 28), interpolation=cv2.INTER_AREA)
        # cv2.imshow('resized_image', resized_img)
        # cv2.waitKey(0)
        return resized_img

    def MultiImageTesting(self, mnist_model_name, mnist_imgs):
        model = tf.keras.models.load_model(mnist_model_name)
        img = np.array(mnist_imgs)
        img = img.reshape(-1, 28, 28, 1)
        input_data = ((np.array(img) / 255) - 1) * -1
        res = np.argmax(model.predict(input_data), axis=-1)
        return res

    def GetFilePath(self):
        path = "Images/Result"
        filelist = []

        for root, dirs, files in os.walk(path):
            for file in files:
                filelist.append(os.path.join(root, file))

        print("\n")
        v = [x for x in filelist if x.endswith(".png")]
        return v