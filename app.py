import base64
import io
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request
from model import get_unet_model

# Initialise
port = int(os.environ.get("PORT", 5000))
app = Flask(__name__)

# Some Usefull functions
def decode_base64(b64_string):
    img_data = base64.b64decode(b64_string)
    pil_img = Image.open(io.BytesIO(img_data))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_BAYER_BG2GRAY)


def load_model(h5_path, img_size=512, channels=1):
    unet = get_unet_model((img_size,img_size,channels))
    unet.load_weights(h5_path)

    return unet 

@app.route('/preict')
def predict_mask():
    pass