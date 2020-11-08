import os
import sys
import matplotlib.pyplot as plt
import cv2
#import numpy as np
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath(os.getcwd())
sys.path.append(ROOT_DIR)  # To find local version of the library
os.chdir(ROOT_DIR)
print(os.path.abspath(os.curdir))
# Import Mask RCNN
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn.config import Config
import mrcnn.model as modellib
from ants import ants

def displayImage (image, modelPath, config):
    #TODO change to for loop for list implementation
    #for img in image:
    tempmax = max(0,image.shape[0],image.shape[1])
    #for img in image:
    tempmin = min(tempmax,image.shape[0],image.shape[1])


    class tempConfig(config.__class__):
        IMAGE_MIN_DIM= tempmin
        IMAGE_MAX_DIM = tempmax
    tempConfig = tempConfig()
    tempConfig.display()

    with tf.device("/gpu:0"):  # /cpu:0 or /gpu:0
        model = modellib.MaskRCNN(mode="inference", model_dir=modelPath,config=tempConfig)

    model.load_weights(modelPath, by_name=True)

    print("==============================================")
    print("============ Running Detection ===============")
    print("==============================================")

    results = model.detect([image], verbose=1)

    print("==============================================")
    print("============ Printing Results  ===============")
    print("==============================================")

    print(results[0])
    print(results[0]['rois'].shape[0])

    print("==============================================")
    print("============ Visualizing Model ===============")
    print("==============================================")

    r = results[0]
    class_names = ['BG','Full Ant','Head','Thorax','Abdomen']
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], title="Predictions")

def detectImage (image, model, config):

    print("==============================================")
    print("============ Running Detection ===============")
    print("==============================================")

    results = model.detect([image], verbose=0)

    print("==============================================")
    print("======= Printing/Returning Results  ==========")
    print("==============================================")
    #print(results)
    # Display results
    r = results[0]

    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                ["BG","Full Ant","Head","Thorax","Abdomen"], r['scores'], title="Predictions")
        #TODO return results and convert to other format: cx, cy, theta

def main():
    print("doing main")
    #ROOT_DIR = os.path.abspath(os.getcwd())
    imagePath = os.path.abspath(r"C:\Users\Max\Documents\Github\ScrantonTracker\Projects\sampleAnt\download.jpg")
    modelPath = os.path.abspath(r"C:\Users\Max\Documents\ScrantonTrackerDataset\models\TRAINEDFULLANTS824.h5")
    #r"/home/ant/ScrANTontracker/logs/TRAINEDFULLANTS824.h5"
    if not os.path.exists(modelPath):
        print("Model at ",modelPath," does not exist.")
    if not os.path.exists(imagePath):
        print("Image at ",imagePath," does not exist.")
    #Inherit config from mrcnn.config.Config
    #Image dimensions will be changed within functions
    class InferenceConfig(Config):
        NAME = "ScrANTonData"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 4
        DETECTION_MIN_CONFIDENCE = .1
        IMAGE_RESIZE_MODE = "square"
        IMAGE_MAX_DIM = 256
        IMAGE_MAX_DIM = 256

    config = InferenceConfig()

    print("loading image from",imagePath)
    image = cv2.imread(imagePath)

    with tf.device("/gpu:0"):  # /cpu:0 or /gpu:0
        model = modellib.MaskRCNN(mode="inference", model_dir=modelPath,config=config)

    model.load_weights(modelPath, by_name=True)

    mode = 'detect'
    if image is not None:
        if mode == 'display':
            print("============== Display Mode ==================")
            displayImage(image, model, config)
        if mode == 'detect':
            print("=============== Detect Mode ==================")
            detectImage(image, model, config)
    else:
        print("Iamge is None")

if __name__ == "__main__":
    main()
