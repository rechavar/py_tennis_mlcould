import base64
import io
import os
import cv2
import numpy as np
from PIL import Image
import json
from tensorflow.keras.models import load_model

# Some Usefull functions
def decode_base64(b64_string):
    img_data = base64.b64decode(b64_string)
    pil_img = Image.open(io.BytesIO(img_data))
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_BAYER_BG2GRAY)
    return cv2.resize(img,(512, 512), interpolation= cv2.INTER_AREA )

def encode_base64(img):
    return base64.b64encode(img)


def init():
    global unet
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'model-dsbowl2018-1.h5')
    unet = load_model(model_path)


def run(raw_data):
    b64_str = np.array(json.loads(raw_data)['data'])

    img = decode_base64(b64_str)

    prediction = unet.predict(img)

    return encode_base64(prediction)

