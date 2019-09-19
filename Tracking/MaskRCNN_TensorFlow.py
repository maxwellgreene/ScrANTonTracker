import os
import sys
import random
import skimage.io
from skimage import img_as_ubyte
from skimage import img_as_float
import colorsys
import numpy as np

import time
import glob
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../Mask_RCNN-master")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import carpenter configs
import Training.train_carpenter as carpConfigs

def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

class InferenceConfig(carpConfigs.CarpenterConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.3


class MaskRCNN_TensorFlow:
    def __init__(self):
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        MODEL_PATH = 'mask_rcnn_carpenter_0250.h5'
        
        config = InferenceConfig()
        #self.config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        weights_path = "/home/intelalert/IntelAlertPython2/mask_rcnn_carpenter_0250.h5" 


        # Load weights trained on MS-COCO
        #model.load_weights(MODEL_PATH, by_name=True)
        self.model.load_weights(weights_path, by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        class_names = ['bg', 'person','car', 'truck']
        self.labels ={}
        for i, name, in enumerate(class_names):
            self.labels[i] = name

    def find_items(self, a_file_name):
       
        try:
            image = skimage.io.imread(a_file_name)
        except:
            try:
                image = cv2.imread(a_file_name)
                cv2.imwrite(image)
            except:    
                return None, None
        
        
        results = self.model.detect([image], verbose=1)
        r = results[0]

        # convert image to open_cv so it works with what we have already done
        cv_image = img_as_ubyte(image)

        #reformate the results and return them
        return self.reformat(r['rois'],r['masks'], r['scores'], r['class_ids']), cv_image[:,:,::-1]


    def reformat(self, boxes, masks, scores, class_ids):
        boxes_out = []
        i=0
        for box, score, class_id in zip(boxes, scores, class_ids):
            #print(box)
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY
            boxes_out.append([self.labels[class_id], score, [startX, startY, boxW, boxH], masks[:,:,i]])
            i+=1
        return boxes_out


    def make_result_image(self, clone, a_results,  a_out_file):
        N = len(a_results)
        colors = random_colors(N)
        # loop over the number of detected objects
        #clone = a_image.copy()
        for result, color in zip(a_results, colors):
            (startX, startY, boxW, boxH) = result[2]
            endX = startX + boxW
            endY = startY + boxH

            mask = result[3]
    
            for c in range(3):
                clone[:,:,c] = np.where(mask==1, clone[:,:,c]*.5 + .5 * color[c] *255, clone[:,:,c])
                
            cv2.rectangle(clone, (startY, startX), (endY, endX), color, 2)

            # draw the predicted label and associated probability of the
            # instance segmentation on the image
            text = "{}: {:.4f}".format(result[0], result[1])
            self.write_on_image(clone, text, (startY, startX - 5), 3)

        # todo remove writing here as it should happen latter.
        # cv2.imwrite(a_out_file, clone)

    def write_on_image(self, image, text_str, location, size):
        cv2.putText(image,
                    text_str,
                    location,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=size,
                    color=(0, 0, 255))

