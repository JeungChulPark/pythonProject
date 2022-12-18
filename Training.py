import os
import cv2
#import tensorflow.compat.v1 as tf
#import tensorflow.compat.v1.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import onnx
# import keras2onnx
import pandas as pd

from tensorflow.keras.layers import Dense, Conv2D, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input
from tensorflow.keras.layers import Flatten, add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from model.ResNet import ResNet

import math

def Training():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    tf.random.set_seed(1234)

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    version = 3
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, input_shape=(28,28,1), padding='same', activation='relu'),
        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=64, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=128, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=256, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=512, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(kernel_size=(3, 3), filters=1024, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1024, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=(x_test, y_test))
    result = model.evaluate(x_test, y_test)

    model.save('mnist_model_modified_epoch10')
    # onnx_model = keras2onnx.convert_keras(model, name="mnist_keras_model", target_opset=9, channel_first_inputs=None)
    # onnx.save_model(onnx_model, "mnist_keras_model.onnx")
    # tf.saved_model.save(model, "mnist_tf_v1_model")

    print("최종 예측 성공률(%): ", result[1]*100)
    return model

def TrainingTransferLearning(xn_train, yn_train):

    model = tf.keras.models.load_model('mnist_model_modified_epoch100')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = np.array(xn_train)
    y_train = np.array(yn_train)

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model.compile(loss='categorical_crossentropy', optimizer=tf.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    model.summary()
    model.fit(x_train, y_train, batch_size=100, epochs=100, validation_data=(x_test, y_test))
    result = model.evaluate(x_test, y_test)

    model.save('mnist_model_modified_epoch100_tl')
    # onnx_model = keras2onnx.convert_keras(model, name="mnist_keras_model", target_opset=9, channel_first_inputs=None)
    # onnx.save_model(onnx_model, "mnist_keras_model.onnx")
    # tf.saved_model.save(model, "mnist_tf_v1_model")

    print("최종 예측 성공률(%): ", result[1]*100)
    return model

def ExtractImage(filename):
    img = cv2.imread(filename)
    #plt.figure(figsize=(15,12))
    #print("img")

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (5,5),0)

    ret, img_th = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierachy = cv2.findContours(img_th.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    rects = [cv2.boundingRect(each) for each in contours]
    #print(rects)
    rects = sorted(rects)

    # thickness = abs(rects[0][2]-rects[1][2])*2
    #
    # contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    # biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
    #
    # cv2.drawContours(img_blur, biggest_contour,-1, (255,255,255), thickness)
#    cv2.imshow('217_blur', img_blur)
#    cv2.waitKey(0)

    # ret, img_th = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)
    # contours, hierachy = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    # rects = [cv2.boundingRect(each) for each in contours]
    # rects = sorted(rects)

    # cv2.imshow('217_blur2', img_blur)
    # cv2.waitKey(0)

    img_for_class = img_blur.copy()
    mnist_imgs=[]
    margin_pixel = 15

    i = 0
    for rect in rects:
        # print(rect)
        # print(rect[1] - margin_pixel, rect[1] + rect[3] + margin_pixel, rect[0] - margin_pixel, rect[0] + rect[
        #     2] + margin_pixel)
        im=img_for_class[rect[1]-margin_pixel:rect[1]+rect[3]+margin_pixel, rect[0]-margin_pixel:rect[0]+rect[2]+margin_pixel]
        # cv2.imshow('im', im)
        # cv2.waitKey(0)

        row, col = im.shape[:2]
        #print("row", row, col)
        bordersize = max(row, col)
        diff = min(row, col)

        #print('bordersize', bordersize, diff)

        bottom = im[row-2:row, 0:col]
        mean = cv2.mean(bottom)[0]
        #print(mean)
        border = cv2.copyMakeBorder(
            im,
            top=0,
            bottom=0,
            left=int((bordersize-diff)/2),
            right=int((bordersize-diff)/2),
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean]
        )

        # cv2.imshow('border_image', border)
        # cv2.waitKey(0)

        square = border
        #cv2.imshow('square', square)

        resized_img = cv2.resize(square, dsize=(28,28), interpolation=cv2.INTER_AREA)
        mnist_imgs.append(resized_img)
        # cv2.imshow('resized_image', resized_img)
        # name = str(i)+'.png'
        # cv2.imwrite(name, resized_img)
        # cv2.waitKey(0)
        i += 1

    return mnist_imgs

def ResizeImage(filename):
    img = cv2.imread(filename)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_blur = cv2.GaussianBlur(img_gray, (5,5),0)

    ret, img_th = cv2.threshold(img_blur, 127, 255, cv2.THRESH_BINARY_INV)

    resized_img = cv2.resize(img_th, dsize=(28,28), interpolation=cv2.INTER_AREA)
    # cv2.imshow('resized_image', resized_img)
    # cv2.waitKey(0)
    return resized_img

def MultiImageTesting(mnist_model_name, mnist_imgs):
    model = tf.keras.models.load_model(mnist_model_name)
    img = np.array(mnist_imgs)
    img=img.reshape(-1, 28, 28, 1)
    input_data = ((np.array(img)/255)-1)*-1
    res = np.argmax(model.predict(input_data), axis=-1)
    return res

def SingleImageTesting(f, mnist_imgs):
    model = tf.keras.models.load_model('mnist_model_epoch100_tl')
    img = mnist_imgs
    img = img.reshape(-1, 28, 28, 1)
    input_data = ((np.array(img)/255)-1)*-1
    res = np.argmax(model.predict(input_data), axis=-1)
    print(res)
    f.write("%d\n" % res)

def GetFilePath():
    path = "Image/Result"
    filelist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(root, file))

    print("\n")
    v = [x for x in filelist if x.endswith(".png")]
    return v

# model = Training()

"""
subject_1, subject_2, subject_3에서 정답인 것을 다시 학습시키는 모듈
"""
def MNISTTransferLearning(inference_model_name=''):
    v = GetFilePath()
    img_array = []
    img_ans = []
    for file in v:
        strings = file.split('\\')
        if strings[1] == "subject_4":
            break
        mnist_imgs = ResizeImage(file)
        img_array.append(mnist_imgs)

    res = MultiImageTesting(inference_model_name, img_array)

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
        i = i+1
    return img_array_tl, img_array_tl_res


""" 
inferencing for transfer learning 
"""
def InferencingMNISTTransforLearning():
    v = GetFilePath()
    img_array = []
    for file in v:
        mnist_imgs = ResizeImage(file)
        img_array.append(mnist_imgs)
    result_mnist_model = MultiImageTesting('mnist_model_modified_epoch100', img_array)

    for file in v:
        mnist_imgs = ResizeImage(file)
        img_array.append(mnist_imgs)
    res = MultiImageTesting('mnist_model_modified_epoch100_tl', img_array)
    print(res)

    f = open("Images/Result/answer_modified_epoch100_tl.txt", 'w')
    i = 0
    for file in v:
        strings = file.split('\\')
        f.write(strings[-5] + " " + strings[-4] + " " + strings[-3] + " " + strings[-2] + " " + strings[-1] + " ")
        f.write("%d %d %d " % (int(strings[-2]), int(result_mnist_model[i]), int(res[i])))
        if int(strings[-2]) == int(result_mnist_model[i]):
            f.write("%d " % 1)
        else:
            f.write("%d " % 0)

        if int(strings[-2]) == int(res[i]):
            f.write("%d\n" % 1)
        else:
            f.write("%d\n" % 0)
        i = i+1
    f.close()

# Training()

# img_array_tl, img_array_tl_res = MNISTTransferLearning('mnist_model_modified_epoch100')
# TrainingTransferLearning(img_array_tl, img_array_tl_res)

# TrainingResNet(1)
# TrainingResNetTransferLearning(2, 'saved_models/mnist_ResNet29v2_model.093.h5')

resnet = ResNet()
resnet.Train(1)

# v = GetFilePath()
# img_array = []
# print('inferencing : ')
# print("%d" % len(v))
# for file in v:
#     mnist_imgs = ResizeImage(file)
#     img_array.append(mnist_imgs)
# result_mnist_model = MultiImageTesting('mnist_model_modified_epoch10', img_array)
# print(result_mnist_model)
#
# f = open("Image/Result/answer_epoch10.txt", 'w')
# for result in result_mnist_model:
#     f.write("%d\n" % result)
# f.close()

