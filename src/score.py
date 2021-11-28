import base64
import io
import os
import cv2
import numpy as np
from PIL import Image
import json
from tensorflow.keras.models import load_model
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse


import tensorflow as tf
from tensorflow.keras.layers import * 

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def get_unet_model(input_shape, num_classes):

    inputs = Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(num_classes, 1, padding="same", activation="softmax")(d4)

    model = tf.keras.Model(inputs, outputs, name="U-Net")

    return model

# Some Usefull functions
def prep_img(image):
    img = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2GRAY)
    riseze_img = cv2.resize(img,(512, 512), interpolation= cv2.INTER_AREA)

    return np.stack(riseze_img)


def decode_prediction(pred):
  mask = np.zeros((pred.shape[0], pred.shape[0]))

  for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):

      mask[i,j] = np.argmax(pred[i,j,:])

  return mask


def init():
    global unet
    unet = get_unet_model((512,512,1),5)
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'model-dsbowl2018-1.h5')
    unet.load_weights(model_path)

@rawhttp
def run(request):
    if request.method == 'POST':
        file_bytes = request.files["image"]
        image = Image.open(file_bytes).convert('RGB')
        img_ready = np.asarray(image, dtype=np.int)

        prediction = unet.predict(np.stack(img_ready))

        mask = decode_prediction(prediction)
        
        return AMLResponse(json.dumps(mask), 200)

