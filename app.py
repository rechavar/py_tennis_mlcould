import base64
import io
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request

# Initialise
port = int(os.environ.get("PORT", 5000))
app = Flask(__name__)

# Some Usefull functions
def decode_base64(b64_string):
    image_64_decode = base64.decodestring(bytes(b64_string, "UTF-8")) 
    image_result = open('deer_decode_test.png', 'wb') # create a writable image and write the decoding result
    image_result.write(image_64_decode)
    return "Holi"

def encode_base64(img):
    return base64.b64encode(img)

@app.route('/preict', methods=['POST'])
def predict_mask():
    data = request.json['data']

    img = decode_base64(data)

    cv2.imwrite('chatched_img.png', img)


if __name__ == '__main__':
    app.run()