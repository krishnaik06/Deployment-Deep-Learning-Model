from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import PIL.Image
import PIL.ImageOps as ImageOps
import matplotlib.pyplot as plt
from skimage.transform import resize

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.models import model_from_json
json_file = open(r'C:\Users\rishi\deployemotion\depemotion\Deployment-Deep-Learning-Model\models\fer.json', 'r')
loaded_model_json = json_file.read()
model = model_from_json(loaded_model_json)
model.load_weights(r'C:\Users\rishi\deployemotion\depemotion\Deployment-Deep-Learning-Model\models\fer.h5')

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()


# Load your trained model

          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img=plt.imread(img_path)
    img2=resize(img,(48,48))
    rgb_weights = [0.2989, 0.5870, 0.1140]

    imgf = np.dot(img2[...,:3], rgb_weights)
    imgf=imgf.reshape(1,48,48,1)

    # img = image.load_img(img_path, target_size=(48,48))
    # img=ImageOps.grayscale(img)
    
    # # Preprocessing the image
    # x = image.img_to_array(img)
    # # x = np.true_divide(x, 255)
    # x = np.expand_dims(x, axis=0)
    # # print(x.shape())

    # # Be careful how your trained model deals with the input
    # # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='caffe')

    preds = model.predict(imgf)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        
        emotions=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad','Surprise','Neutral']
        preds=list(preds)
        
               # ImageNet Decode
        maxelem=max(preds[0])
        i=0
        for x in preds[0]:
            if maxelem==x:
                break
            i+=1   
        return str(emotions[i])
    return None


if __name__ == '__main__':
    app.run(debug=True)

