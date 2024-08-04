'''
Ref: https://keras.io/examples/vision/grad_cam/
Purpose: For visualizing XAI (what region did you saw?)
Writer: SungwookLE
Date: '21.10/18
'''
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
import cv2

class explainable_model:
    def __init__(self, model):
        self.model_load = model
        self.model_load.layers[-1].activation = None
        #print(self.model_load.summary())

    def explainable_model(self, img, last_conv_layer_name, alpha=0.4, output_node = None):
        self.last_conv_layer_name =last_conv_layer_name

        '''
        아래는 이미지 로더로 부터 데이터 받는 경우에 노말라이즈 포함해서 데이터 전처리해주는 과정
        
        idx = random.randint(0, len(X_my)-1)
        img_array = get_img_array(X_my[idx], (160,120))
        print(img_array.shape)
        plt.imshow(X_my[idx])
        '''
        array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(array, axis=0)
       
        # Generate class activation heatmap
        heatmap = self.make_gradcam_heatmap(img_array, self.model_load, self.last_conv_layer_name, output_node=output_node)
        superimposed_img = self.save_and_display_gradcam(img*255.0, heatmap, alpha=alpha)

        fig , (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
        # Display heatmap
        ax1.matshow(heatmap)
        #plt.colorbar()
        ax2.imshow(superimposed_img)
        return heatmap


    def get_img_array(self, img, dsize):

        img = cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_AREA)
        img = img/255.0
        array = keras.preprocessing.image.img_to_array(img)
        array = np.expand_dims(array, axis=0)
        return array

    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None, output_node = None):
        # First, we create a model that maps the input image to the activations
        # of the last conv layer as well as the output predictions

        if output_node is None:
            grad_model = tf.keras.models.Model(
                [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
            )
        else:
            grad_model = tf.keras.models.Model(
                [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output[output_node]]
            )

        # Then, we compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        # This is the gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)

        # This is a vector where each entry is the mean intensity of the gradient
        # over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        # then sum all the channels to obtain the heatmap class activation
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # For visualization purpose, we will also normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()


    def save_and_display_gradcam(self, img, heatmap, cam_path="cam.jpg", alpha=0.4):
        # Load the original image
    
        img = keras.preprocessing.image.img_to_array(img)

        # Rescale heatmap to a range 0-255
        heatmap = np.uint8(255 * heatmap)

        # Use jet colormap to colorize heatmap
        jet = cm.get_cmap("jet")

        # Use RGB values of the colormap
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]

        # Create an image with RGB colorized heatmap
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

        # Superimpose the heatmap on original image
        superimposed_img = jet_heatmap * alpha + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

        # Save the superimposed image
        superimposed_img.save(cam_path)

        # Display Grad CAM
        #plt.imshow(superimposed_img)
        return superimposed_img