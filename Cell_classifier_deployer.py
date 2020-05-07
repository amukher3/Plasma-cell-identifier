# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 21:50:18 2020

@author:Abhishek Mukherjee
"""

import os
from flask import Flask, request, jsonify, url_for, render_template
from tensorflow.keras.models import load_model
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image


ALLOWED_EXTENSION  =set(['txt', 'pdf', 'png','jpg','jpeg','gif','bmp'])
IMAGE_HEIGHT =64
IMAGE_WIDTH = 64
IMAGE_CHANNELS = 3
os.chdir(r'C:\Users\abhi0\Downloads\flaskdata')
app = Flask(__name__)
model = load_model('C:/Users/abhi0/OneDrive/Documents/BloodPlasmaIdentification_new/best_model.h5')

@app.route('/')
def index():
    return render_template('ImageML.html')

@app.route('/api/image', methods=['POST'])
def upload_image():
    file = request.files['image']
    x = []
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    img = Image.open(BytesIO(file.read()))
    img.load()
    img  = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
    x  = image.img_to_array(img)
    test_image = x.reshape((1,64,64,3))
    pred = model.predict_classes(test_image)
    
    if pred==1:
        return render_template('ImageML.html', prediction = 'I would say the image is most likely a Normal Cell')
    else:
        return render_template('ImageML.html', prediction = 'I would say the image is most likely a Blast Cell')
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
    
                 