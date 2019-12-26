from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pandas as pd
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from werkzeug.utils import secure_filename

IMG_HEIGHT = 250
IMG_WIDTH = 250
UPLOAD_FOLDER = '/Users/sean/wtd/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class ModelRequester(object):
    def __init__(self):
        self.model = tf.keras.models.load_model('wtd_model.h5')

    def get_labels(self, img_arr):
        self.confidence = self.model.predict(img_arr)
        output = np.argmax(self.confidence, axis=1)
        self.confidence = np.max(self.confidence, axis=1)[0]
        return output

    def do_prediction(self, filename):
        self.test_image = image.load_img(filename, target_size=[IMG_WIDTH, IMG_HEIGHT])
        self.test_image = image.img_to_array(self.test_image)
        self.test_image = np.expand_dims(self.test_image, axis=0)
        output = self.get_labels(self.test_image)[0]
        self.last_output = output
        return self.get_label(output)

    def get_caption(self):
        print(self.get_label(self.last_output))
        print(self.confidence)
        if self.last_output == 0:
            duck_str = "is a duck"
        else:
            duck_str = "is not a duck"
        return "I'm guessing this {} (I'm {}% sure)".format(duck_str, self.confidence*100)
    
    def get_label(self, output):
        if output == 0:
            return 'duck'
        else:
            return 'notduck'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def do_GET(request):
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

def do_POST(request):
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(full_filename)
        mr = ModelRequester()
        label = mr.do_prediction(full_filename)
        caption = mr.get_caption()
        return render_template("index.html", user_image = filename, label=label, caption=caption)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        return do_POST(request)
    return do_GET(request)



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
if __name__ == '__main__':
    app.run()
