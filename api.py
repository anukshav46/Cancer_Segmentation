import os
from flask import request
from flask import Flask
from flask import render_template

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, UpSampling2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.applications.xception import Xception

import numpy as np
import tensorflow as tf
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
from tensorflow.keras import backend as K

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import CustomObjectScope
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.metrics import MeanIoU

app=Flask(__name__)
UPLOAD_FOLDER="C:/Users/DELL/Desktop/cancer project/static"
MODEL=None
DEVICE="cuda"
H = 224
W = 224
preds=[]
def Combo_loss(y_true, y_pred, smooth=1):
 
 e = K.epsilon()
 if y_pred.shape[-1] <= 1:
   ALPHA = 0.8    # < 0.5 penalises FP more, > 0.5 penalises FN more
   CE_RATIO = 0.5 # weighted contribution of modified CE loss compared to Dice loss
   y_pred = tf.keras.activations.sigmoid(y_pred)
 elif y_pred.shape[-1] >= 2:
   ALPHA = 0.3    # < 0.5 penalises FP more, > 0.5 penalises FN more
   CE_RATIO = 0.7 # weighted contribution of modified CE loss compared to Dice loss
   y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
   y_true = K.squeeze(y_true, 3)
   y_true = tf.cast(y_true, "int32")
   y_true = tf.one_hot(y_true, num_class, axis=-1)
 
 # cast to float32 datatype
 y_true = K.cast(y_true, 'float32')
 y_pred = K.cast(y_pred, 'float32')
 
 targets = K.flatten(y_true)
 inputs = K.flatten(y_pred)
 
 intersection = K.sum(targets * inputs)
 dice = (2. * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
 inputs = K.clip(inputs, e, 1.0 - e)
 out = - (ALPHA * ((targets * K.log(inputs)) + ((1 - ALPHA) * (1.0 - targets) * K.log(1.0 - inputs))))
 weighted_ce = K.mean(out, axis=-1)
 combo = (CE_RATIO * weighted_ce) - ((1 - CE_RATIO) * dice)
 
 return combo

def test(model,result_path,image):

    """ Prediction and Evaluation """
    name=image.split("\\")[-1]
    img = cv2.imread(image, cv2.IMREAD_COLOR) ## [H, w, 3]
    img = cv2.resize(img, (W, H))       ## [H, w, 3]
    x = img/255.0                         ## [H, w, 3]
    x = np.expand_dims(x, axis=0)           ## [1, H, w, 3]


    """ Prediction """
    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)

    """ Saving the prediction """

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255
    preds.append(y_pred)
    save_image_path = os.path.join(result_path, name)
    print(save_image_path)
    cv2.imwrite(save_image_path,y_pred)
            #y_pred.save(image_location)
   # cv2.imwrite(result_path, y_pred)

def ensemble_model(result_path,img):

    with CustomObjectScope({"Combo_loss": Combo_loss}):
      model1 = load_model('C:/Users/DELL/Desktop/cancer project/model1.h5', compile=False)
      model2 = load_model('C:/Users/DELL/Desktop/cancer project/model2.h5', compile=False)
      model3 = load_model('C:/Users/DELL/Desktop/cancer project/model3.h5', compile=False)
    
    models = [model1,model2,model3]
    weights = [0.5, 0.3, 0.2]

    """ Prediction and Evaluation """
    SCORE = []
    """ Extracting the name """
    name = img.split("\\")[-1]
    print(name)
    """ Reading the image """
    image = cv2.imread(img, cv2.IMREAD_COLOR) ## [H, w, 3]
    image = cv2.resize(image, (W, H))       ## [H, w, 3]
    x = image/255.0                         ## [H, w, 3]
    x = np.expand_dims(x, axis=0)           ## [1, H, w, 3]

    """ Reading the mask """
    # mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
    # mask = cv2.resize(mask, (W, H))

    """ Prediction """
    preds = [model.predict(x,verbose=0)[0] for model in models]
    preds=np.array(preds)
    y_pred = np.tensordot(preds, weights, axes=((0),(0)))
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = y_pred >= 0.5
    y_pred = y_pred.astype(np.int32)
    print(y_pred)
    """ Saving the prediction """
    save_image_path = os.path.join(result_path, name)
    print(save_image_path)
  
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    line = np.ones((H, 10, 3)) * 255
   
    cv2.imwrite(save_image_path,y_pred)
    return name
    #save_results(y_pred, save_image_path)
    
    """ Flatten the array """
    # mask = mask/255.0
    # mask = (mask > 0.5).astype(np.int32).flatten()
    # y_pred = y_pred.flatten()

@app.route("/",methods=["GET","POST"])
def upload_predict():
    if request.method=="POST":
        image_file=request.files["image"]
        if image_file:
            image_location=os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
            image_file.save(image_location)
            result_path="C:/Users/DELL/Desktop/cancer project/static/ensemble"
            result = ensemble_model(result_path, image_location)           
            return render_template("index.html", image_loc=image_file.filename, result_loc=result)
    return render_template("index.html", image_loc=None, result_loc=None)

if __name__ == "__main__":
        
    app.run(port=12000,debug=True)

