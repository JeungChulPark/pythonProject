import tf2onnx
import tensorflow.compat.v1.keras as keras

net = keras.models.load_model('mnist_keras_v1_model.h5')

tf2onnx.convert.from_keras(
    net,
    opset=9,
    output_path='mnist_keras_v1_model.onnx'
)