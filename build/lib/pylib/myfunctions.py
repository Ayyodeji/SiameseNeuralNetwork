import os
import cv2
import time
import random
import pandas as pd
import math
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from keras import backend, layers, metrics
from keras.optimizers import Adam
from keras.applications import Xception, ResNet50, DenseNet121, InceptionV3, DenseNet201
from keras.models import Model, Sequential

from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
def get_encoder(input_shape):
    """ Returns the image encoding model """

    # RESNET50
    pretrained_model = ResNet50(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False,
        pooling='avg',
    )
    for i in range(len(pretrained_model.layers)-27):
        pretrained_model.layers[i].trainable = False

    encode_model = Sequential([
        pretrained_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))
    ], name="Encode_Model")
    return encode_model
    
class DistanceLayer(layers.Layer):
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)
    

def get_siamese_network(input_shape = (128, 128, 3)):
    encoder = get_encoder(input_shape)
    
    # Input Layers for the images
    anchor_input   = layers.Input(input_shape, name="Anchor_Input")
    positive_input = layers.Input(input_shape, name="Positive_Input")
    negative_input = layers.Input(input_shape, name="Negative_Input")
    
    ## Generate the encodings (feature vectors) for the images
    encoded_a = encoder(anchor_input)
    encoded_p = encoder(positive_input)
    encoded_n = encoder(negative_input)
    
    # A layer to compute ‖f(A) - f(P)‖² and ‖f(A) - f(N)‖²
    distances = DistanceLayer()(
        encoder(anchor_input),
        encoder(positive_input),
        encoder(negative_input)
    )
    
    # Creating the Model
    siamese_network = Model(
        inputs  = [anchor_input, positive_input, negative_input],
        outputs = distances,
        name = "Siamese_Network"
    )
    return siamese_network

siamese_network = get_siamese_network()
class SiameseModel(Model):
    # Builds a Siamese model based on a base-model
    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()
        
        self.margin = margin
        self.siamese_network = siamese_network
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape get the gradients when we compute loss, and uses them to update the weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)
            
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.siamese_network.trainable_weights))
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # Get the two distances from the network, then compute the triplet loss
        ap_distance, an_distance = self.siamese_network(data)
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self):
        # We need to list our metrics so the reset_states() can be called automatically.
        return [self.loss_tracker]
siamese_model = SiameseModel(siamese_network)

optimizer = Adam(learning_rate=1e-3, epsilon=1e-01)
siamese_model.compile(optimizer=optimizer)
siamese_model.load_weights("/content/drive/My Drive/siamese_model")

def extract_encoder(model):
    encoder = get_encoder((128, 128, 3))
    i=0
    for e_layer in model.layers[0].layers[3].layers:
        layer_weight = e_layer.get_weights()
        encoder.layers[i].set_weights(layer_weight)
        i+=1
    return encoder

encoder = extract_encoder(siamese_model)
encoder.save_weights("encoder")

def predict_similarity(image1, image2, threshold=0.8):
    img1 = []
    img2 = []
    image_1 = cv2.imread(image1)
    image_2 = cv2.imread(image2)

    image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
    image_1 = cv2.resize(image_1, (128,128))
    
    image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
    image_2 = cv2.resize(image_2, (128,128))

    img1.append(image_1)
    img2.append(image_2)
    img1 = preprocess_input(np.array(img1))
    img2 = preprocess_input(np.array(img2))
    tensor1 = encoder.predict(img1)
    tensor2 = encoder.predict(img2)

    distance = np.sum(tensor1*tensor2, axis=1)/(norm(tensor1, axis=1)*norm(tensor2, axis=1))
    prediction = np.where(distance<=threshold, 0, 1)
    distance = distance*100
    return str(str(distance[0]) + '%')
def bulktest(img, dirlist):
  arrBT = []
  arrBTN = []
  for cls in os.listdir(dirlist):        
        arrBTN.append(str(img) + '   vs   ' + str(cls).split(".")[0]) 
        arrBT.append(predict_similarity( os.path.join(str(img)), os.path.join(dirlist,cls)))

  df = pd.DataFrame({'Comparison': arrBTN, 'Result': arrBT})
  df = df.drop_duplicates(subset=['Result'])
  df = df.sort_values('Comparison')
  return df
