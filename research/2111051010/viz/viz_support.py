import cv2
from viz.config import get_parse_args
import random
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os
from tensorflow import keras
import matplotlib.cm as cm
import tensorflow as tf
from utils.process import label_dict_static



class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img

class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print("Video Shape is ({0},{1}), FPS is {2}.".format(width, height, fps))

        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

def demo_init(args):
    net_multi = args.model_multi
    net_belt = args.model_belt


    frame_provider = ImageReader(args.images)
    if args.video != '':
        frame_provider = VideoReader(args.video)
        image_flag = False
    else:
        image_flag = True

    return [net_multi, net_belt], frame_provider


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, output_node=None):
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

def display_gradcam(img, heatmap, alpha=0.4):
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
        superimposed_img = np.array(superimposed_img)
        #superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR)

        return superimposed_img


def explainable(model, img, last_conv_layer_name, alpha =0.4, output_node=None):
    array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(array, axis=0)

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, output_node=output_node)
    superimposed_img = display_gradcam(img*255.0, heatmap, alpha=alpha)

    return superimposed_img


def run_demo(net, frame_provider):
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'MP4V') , 20.0, (480,320))

    label_map_oop, label_str_oop = label_dict_static(classifier="OOP")
    label_map_weak, label_str_weak = label_dict_static(classifier="Weak")
    label_map_mask, label_str_mask = label_dict_static(classifier="Mask")
    label_map_belt, label_str_belt = label_dict_static(classifier="Belt")


    model_load_multi = load_model("./ckpt/"+net[0])
    model_load_belt = load_model("./ckpt/"+net[1])

    for img in frame_provider:
        img_belt= cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_AREA)
        orig_img = img.copy()
        img = cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_AREA)
        img = img/255.0
        pred= model_load_multi.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]))

        img_belt  = img_belt/255.0
        pred_belt =  model_load_belt.predict(img_belt.reshape(1, img_belt.shape[0], img_belt.shape[1], img_belt.shape[2]))

        print("Pred label(OOP/Weak/Mask/Belt): ", (np.argmax(pred[0]),np.argmax(pred[1]),np.argmax(pred[2]), np.argmax(pred_belt)))
        img1 = cv2.resize(orig_img, dsize=(320, 320), interpolation=cv2.INTER_AREA)

        for key, val in label_map_oop.items():
            if val == (np.argmax(pred[0])):
                prob = np.exp(np.max(pred[0])) / np.sum(np.exp(pred[0]))
                cv2.putText(img1, "{:<4s}".format(label_str_oop[key])+": {:0.4f}".format(prob), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,128,0), 2)
        
        for key, val in label_map_weak.items():
            if val == (np.argmax(pred[1])):
                prob = np.max(pred[1])
                cv2.putText(img1, "{:<4s}".format(label_str_weak[key])+": {:0.4f}".format(prob), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128,0,0), 2)
        
        for key, val in label_map_mask.items():
            if val == (np.argmax(pred[2])):
                prob = np.exp(np.max(pred[2])) / np.sum(np.exp(pred[2]))
                cv2.putText(img1, "{:<4s}".format(label_str_mask[key])+": {0:.4f}".format(prob), (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,128), 2)

        for key, val in label_map_belt.items():
            if val == (np.argmax(pred_belt)):
                prob = np.exp(np.max(pred_belt)) / np.sum(np.exp(pred_belt))
                cv2.putText(img1, "{:<4s}".format(label_str_belt[key])+": {0:.4f}".format(prob), (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        img2 = explainable(model_load_multi, img, "dropout_8", alpha=0.4, output_node=0)
        img2 = cv2.resize(img2, dsize=(160, 160), interpolation=cv2.INTER_AREA)
        cv2.putText(img2, "xAI(OOP/Weak/Mask)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)

        img3= explainable(model_load_belt, img_belt, "3rd_maxpool", alpha=0.4)
        img3 = cv2.resize(img3, dsize=(160, 160), interpolation=cv2.INTER_AREA)
        cv2.putText(img3, "xAI(Belt)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)

        img_list_v = [img2, img3]
        img_v = cv2.vconcat(img_list_v)

        img_list_h = [img1, img_v]
        img_h = cv2.hconcat(img_list_h)
        cv2.imshow("Multi Classifier", img_h)
        out.write(img_h)


        key = cv2.waitKey(1)

        if key == 27:  # esc
            break
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1
    return