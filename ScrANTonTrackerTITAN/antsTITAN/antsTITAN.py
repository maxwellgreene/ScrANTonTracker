"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import urllib.request
import numpy as np
import skimage.draw
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath(r"/home/antlover/Documents/ScrANTonTrackerTITAN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

#Import COCO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, r"models/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class AntConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "ants"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 4

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 40

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class AntDataset(utils.Dataset):

    def load_ant(self, dataset_dir, subset):
        
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        # Add classes. We have only one class to add.
#        self.add_class("Full Ant", 1, "Full Ant")
#        self.add_class("ScrANTon",4,"Head")
#        self.add_class("ScrANton",3,"Thorax")
#        self.add_class("ScrANton",2,"Abdomen")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, "train")

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
            
#        annotations = json.load(open(os.path.join(dataset_dir, "antsJSON.json")))
#        annotations = [a for a in annotations if type(a['Label']) is dict]
#
#        # Add images
#        for a in annotations:
#            temp = a['Label']
#            FullAntPolygons=[r for r in temp['Full Ant']]
##            HeadPolygons=[r for r in temp['Head']]
##            ThoraxPolygons=[r for r in temp['Thorax']]
##            AbdomenPolygons=[r for r in temp['Abdomen']]
#
#            image = Image.open(urllib.request.urlopen(a['Labeled Data']))
#            height = image.height
#            width = image.width
#
#            self.add_image(
#                "Full Ant",
#                image_id=a['ID'],  # use file name as a unique image id
#                path=a['Labeled Data'],
#                width=width, height=height,
#                polygons=FullAntPolygons)
        
        antsCOCO = COCO(os.path.join(dataset_dir, "antsCOCO.json"))
        
        # Add classes
        for i in antsCOCO.getCatIds():
            self.add_class("ScrANTonDataset", i, antsCOCO.loadCats(i)[0]["name"])
        
        # Add images
        for i in antsCOCO.getImgIds():
            if antsCOCO.loadAnns(antsCOCO.getAnnIds(imgIds=[i],catIds=antsCOCO.getCatIds())):
                self.add_image(
                "ScrANTonDataset", image_id=i,
                path = antsCOCO.imgs[i]['coco_url'],#image = Image.open(urllib.request.urlopen(antsCOCO.imgs[i]['coco_url'])),
                width=antsCOCO.imgs[i]["width"],
                height=antsCOCO.imgs[i]["height"],
                annotations=antsCOCO.loadAnns(antsCOCO.getAnnIds(imgIds=[i], 
                    catIds=antsCOCO.getCatIds())))
            
            
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        
        #image_info = self.image_info[image_id]
        #if image_info["source"] != "ant":
            #return super(self.__class__, self).load_mask(image_id)



        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
#        info = self.image_info[image_id]
#        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
#                        dtype=np.uint8)
#        #print("THIS IS THE INFO POLYGONS TYPE:"+str(type(info['polygons'])))
#        #print("THIS IS INFO POLYGONS:")
#        #print(info['polygons'])
#        
#        for i,p in enumerate(info["polygons"]):
#            # Get indexes of pixels inside the polygon and set them to 1
#            #print("pTYPE: "+str(type(p))+" pTYPEgeom"+str(type(p['geometry']))+"pTYPEgeom1: "+str(type(p['geometry'][1])))
#            x=[]
#            y=[]            
#            for j,pp in enumerate(p['geometry']):
#                x.append(pp['x'])
#                y.append(pp['y'])
#            xx=np.asarray(x)
#            yy=np.asarray(y)
#            #rr, cc = skimage.draw.polygon(p['geometry']['y'], p['geometry']['x'])
#            rr, cc = skimage.draw.polygon(yy,xx)
#            mask[rr, cc, i] = 1
#        
#        # Return mask, and array of class IDs of each instance. Since we have
#        # one class ID only, we return an array of 1s
#        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = annotation['category_id'] #self.map_source_class_id("{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(AntDataset, self).load_mask(image_id)



    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "ant":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = AntDataset()
    dataset_train.load_ant(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = AntDataset()
    dataset_val.load_ant(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    
#    model.train(dataset_train, dataset_val,
#                learning_rate=config.LEARNING_RATE,
#                epochs=10,
#                layers='heads')
    
    print("Stage 1: Training network heads")                                                                                                                                                                   
    model.train(dataset_train, dataset_val,                                                                                                                                                           
                learning_rate=config.LEARNING_RATE,                                                                                                                                                   
                epochs=100,                                                                                                                                                                           
                layers='heads')                                                                                                                                                                       
 
 
    # Training - Stage 2                                                                                                                                                                               
    # Finetune layers from ResNet stage 4 and up                                                                                                                                                       
    print("Stage 2: Fine tune Resnet stage 4 and up")                                                                                                                                                          
    model.train(dataset_train, dataset_val,                                                                                                                                                           
                learning_rate=config.LEARNING_RATE,                                                                                                                                                   
                epochs=190,                                                                                                                                                                           
                layers='4+')                                                                                                                                                                          
 
    # Training - Stage 3                                                                                                                                                                               
    # Fine tune all layers                                                                                                                                                                             
    print("Stage 3: Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=200,
                layers='all')
    
    model_path = os.path.join(r"/home/antlover/Documents/ScrANTonTrackerTITAN/logs/TRAINEDFULLANTS.h5")
    model.keras_model.save_weights(model_path)
    
    
#def train(model):
#    """Train the model."""
#    # Training dataset.                                                                                                                                                                                
#    dataset_train = AntDataset()
#    dataset_train.load_carpenter(args.dataset, "train")
#    dataset_train.prepare()
# 
# 
#    # Validation dataset                                                                                                                                                                               
#    dataset_val = AntDataset()
#    dataset_val.load_carpenter(args.dataset, "val")
#    dataset_val.prepare()
# 
#    #Image Augmentation                                                                                                                                                                               
#    # Right/Left flip 50% of the time                                                                                                                                                                  
#    #augmentation = imgaug.augmenters.Fliplr(0.5)
# 
#    # *** This training schedule is an example. Update to your needs ***                                                                                                                               
#    # Since we're using a very small dataset, and starting from                                                                                                                                        
#    # COCO trained weights, we don't need to train too long. Also,                                                                                                                                     
#    # no need to train all layers, just the heads should do it.                                                                                                                                        
#    print("Training network heads")                                                                                                                                                                   
#    model.train(dataset_train, dataset_val,                                                                                                                                                           
#                learning_rate=config.LEARNING_RATE,                                                                                                                                                   
#                epochs=190,                                                                                                                                                                           
#                layers='heads')                                                                                                                                                                       
# 
# 
#    # Training - Stage 2                                                                                                                                                                               
#    # Finetune layers from ResNet stage 4 and up                                                                                                                                                       
#    print("Fine tune Resnet stage 4 and up")                                                                                                                                                          
#    model.train(dataset_train, dataset_val,                                                                                                                                                           
#                learning_rate=config.LEARNING_RATE,                                                                                                                                                   
#                epochs=220,                                                                                                                                                                           
#                layers='4+')                                                                                                                                                                          
# 
#    # Training - Stage 3                                                                                                                                                                               
#    # Fine tune all layers                                                                                                                                                                             
#    print("Fine tune all layers")
#    model.train(dataset_train, dataset_val,
#                learning_rate=config.LEARNING_RATE / 10,
#                epochs=200,
#                layers='all')                                                                                                              

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = AntConfig()
    else:
        class InferenceConfig(AntConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
