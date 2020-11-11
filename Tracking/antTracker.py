import warnings
warnings.filterwarnings("ignore")

import skimage
import statistics
import os
import sys
import math
import re
import numpy as np
import cv2
from pprint import pprint
import matplotlib.pyplot as plt
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath(os.getcwd())

# Import Mask RCNN
from tensorflow.python.framework.versions import VERSION
if float(VERSION[0]) == 2:
    sys.path.append(os.path.join(ROOT_DIR,"mrcnn2"))
elif float(VERSION[0]) == 1:
    sys.path.append(os.path.join(ROOT_DIR,"mrcnn1"))

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config

# Import our scripts
import Ant
import Calibration
#import MaskRCNN_TensorFlow
import Dewarp
import antUtils

#This function is useful for viewing the differnt stages of the ant extraction process
def print_im(a_pic):
    #pic2Display = cv2.resize(a_pic, (800, 800))
    cv2.imshow('image', a_pic)
    k = cv2.waitKey(0)

#This function makes a image with the region inside the polygons set to white
def mask_for_polygons(polygons, im_size):
    """Convert a polygon or multipolygon list back to
       an image mask ndarray"""
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    # function to round and convert to int
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    cv2.fillPoly(img_mask, exteriors, 1)
    return img_mask

class ant(antUtils.Filter):
    def __init__(self, dt, a_initialState, framenum = 0):

        self.xk = a_initialState

        self.Pk = np.array([[20000,     0,   0,    0,    0],
                            [    0, 20000,   0,    0,    0],
                            [    0,     0, 300,    0,    0],
                            [    0,     0,   0, 1000,    0],
                            [    0,     0,   0,    0, 1000]])

        self.R = np.array([[100,   0,  0,  0,  0],
                           [  0, 100,  0,  0,  0],
                           [  0,   0, 60, 10, 10],
                           [  0,   0, 10, 80,  0],
                           [  0,   0, 10,  0, 80]])

        self.F = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])

        self.Q = np.array([[8000,  200,  0, 0, 0],
                           [ 200, 8000,  0, 0, 0],
                           [   0,    0, 80, 0, 0],
                           [   0,    0,  0, 4, 0],
                           [   0,    0,  0, 0, 4]])

        self.H = np.array([[1.1, 0, 0, 0, 0],
                           [0, 1.1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])

        self.lengths = []
        self.widths = []
        self.thetas = []
        self.areas = []

        self.state_vars = []
        self.state_covars = []
        self.meas_vars = []
        self.xs = []
        self.ys = []
        self.time = []
        self.add_point(self.xk, framenum)

    def getPoint(self,loc = -1):
        return [self.xs[loc],self.ys[loc],self.thetas[loc],self.lengths[loc],self.widths[loc]]

    def getPrediction(self):
        #print("covars\n",self.state_covars)
        point = self.getPoint()
        try:
            cov = self.state_covars[-1]
        except IndexError as ie:
            return(point)
        point, _ = self.predict(point,cov)
        return(point.tolist())

    def update_extra(self):
        pass
        #self.add_point(self.xk, 0)
        #self.updateF(self.getPoint)

    def get_distance(self, a_point):
        return np.sqrt((self.xs[-1] - a_point[0]) ** 2 + (self.ys[-1] - a_point[1]) ** 2)

    def add_point(self, a_point, a_time):
        #self.xs.append(a_point[0])
        #self.ys.append(a_point[1])
        self.xs.append      (a_point[0])
        self.ys.append      (a_point[1])
        self.thetas.append  (a_point[2])
        self.lengths.append (a_point[3])
        self.widths.append  (a_point[4])
        self.areas.append   (a_point[3]*a_point[4])

        self.time.append(a_time)

    def predictionCorrection(self, a_xp, a_Pp):
        """ add your corretion here if one is needed. """
        while a_xp[2] >= 90:
            a_xp[2] -= 180

        while a_xp[2] < -90:
            a_xp[2] += 180

        return a_xp, a_Pp

    def predictedMeasCorrection(self, a_zp, a_S, a_K):
        """ add your corretion here if one is needed. """
        a_zp, __ = self.predictionCorrection(a_zp, None)
        return a_zp, a_S, a_K

    def correctMeasSubtract(self, a_zdiff, a_zp, a_zk):
        """ add your corretion here if one is needed. """
        if np.abs(a_zdiff[2]) > 90:
            a_zdiff[2] = a_zdiff[2] - np.sign(a_zdiff[2]) * 180
        return a_zdiff

    def CorrectUpdate(self, a_xk, a_Pk):
        """ add your corretion here if one is needed. """
        a_xk, a_Pk = self.predictionCorrection(a_xk, a_Pk)
        self.state_vars.append(a_xk)
        self.state_covars.append(a_Pk)
        return a_xk, a_Pk

# Create a MASKRCNN class with a modified detect function
# that can be passed a list of ants to update
class detectorMaskRCNN(modellib.MaskRCNN):
    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)
    #
    # def detect(self, images, ants, verbose=0):
    #     """
    #     Runs the detection pipeline.
    #
    #     images: List of images, potentially of different sizes.
    #
    #     Returns a list of dicts, one dict per image. The dict contains:
    #     rois: [N, (y1, x1, y2, x2)] detection bounding boxes
    #     class_ids: [N] int class IDs
    #     scores: [N] float probability scores for the class IDs
    #     masks: [H, W, N] instance binary masks
    #     """
    #
    #     assert self.mode == "inference", "Create model in inference mode."
    #     assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"
    #
    #     if verbose:
    #         log("Processing {} images".format(len(images)))
    #         for image in images:
    #             log("image", image)
    #
    #     # Mold inputs to format expected by the neural network
    #     molded_images, image_metas, windows = self.mold_inputs(images)
    #     if debugPrint: print("windows1: ",windows)
    #
    #     # Validate image sizes
    #     # All images in a batch MUST be of the same size
    #     image_shape = molded_images[0].shape
    #     for g in molded_images[1:]:
    #         assert g.shape == image_shape,\
    #             "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."
    #     if debugPrint: print("windows2: ",windows)
    #
    #     # Anchors
    #     anchors = self.get_anchors(image_shape)
    #     # Duplicate across the batch dimension because Keras requires it
    #     anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
    #
    #     if verbose:
    #         log("molded_images", molded_images)
    #         log("image_metas", image_metas)
    #         log("anchors", anchors)
    #     # Run object detection
    #     detections, _, _, mrcnn_mask, _, _, _ =\
    #         self.keras_model.predict([molded_images, image_metas, anchors], verbose=2)
    #
    #     # Process detections
    #     if debugPrint: print("windows3: ",windows)
    #     results = []
    #     for i, image in enumerate(images):
    #         final_rois, final_class_ids, final_scores, final_masks =\
    #             self.unmold_detections(detections[i], mrcnn_mask[i],
    #                                    image.shape, molded_images[i].shape,
    #                                    windows[i])
    #         if debugPrint: print("final_rois: ",final_rois)
    #         results.append({
    #             "rois": final_rois,
    #             "class_ids": final_class_ids,
    #             "scores": final_scores,
    #             "masks": final_masks,
    #         })
    #         print(results)
    #     if debugPrint: print("windows4: ",windows)
    #     return results

# Create a config to use with the extractorMaskRCNN class
class extractorConfig(Ant.AntConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.3

class extractorMaskRCNN(antUtils.MaskRCNN_TensorFlow):
    def __init__(self,a_model_dir = None,a_weights_path = None):
        # Directory to save logs and trained model
        if a_model_dir: model_dir = a_model_dir
        else:           model_dir = os.path.join(ROOT_DIR, "logs")

        if a_weights_path:  WEIGHTS_PATH = a_weights_path
        else:               WEIGHTS_PATH = os.path.join(ROOT_DIR,'models/TRAINEDFULLANTS824.h5')

        # self.config = InferenceConfig()
        self.config = extractorConfig()
        #self.config.display()

        # Create model object in inference mode.
        # self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)
        self.model = detectorMaskRCNN(mode="inference", model_dir=model_dir, config=self.config)
        #model.load_weights(MODEL_PATH, by_name=True)
        self.model.load_weights(WEIGHTS_PATH, by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG','Full Ant', 'Head','Thorax', 'Abdomen']
        self.labels = {}
        for i, name, in enumerate(self.class_names):
            self.labels[i] = name

    def findAnts(self, image, ants):

        results = self.model.detect([image], verbose=0)
        r = results[0]
        """ Input:
        a list of dicts, one dict per image. The dict contains:

        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        antLocs = []
        class_ids = ['BG', 'Full Ant', 'Abdomen', 'Thorax', 'Head']

        for i in range((r['masks'].shape[2])):
            mask = r['masks'][:,:,i]
            contours, _ = cv2.findContours(mask.astype('uint8'), mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
            cnt = contours[0]
            if cnt.shape[0] > 5:
                ellipse = cv2.fitEllipse(cnt)
                antLocs.append({
                    "type": str(class_ids[r['class_ids'][i]]),
                    "head": [ellipse[0][0],ellipse[0][1],ellipse[2],ellipse[1][1],ellipse[1][0]]
                })
        ##########################
        """
        Output:
        Headings, a list of dicts, each of the form:
            {
            "type": class id (str),
            "head": [cx,cy,thetas,ls,ws]
            }
        """
        #if np.random.choice(a=[True,False],size=(1,1),p=[.01,.99]):
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_ids, r['scores'], title="Predictions")
        vis = None

        #should return a list of the form: [cx, cy, theta, l, w]
        return vis, antLocs

class antTracker:
    def __init__(self):
        self.dt = 1
        self.extractor = extractorMaskRCNN()
        self.ants = []
        self.frames = []

        self.colors = ['r', 'g', 'b', 'k', 'm']

    def setup(self, a_vidName, a_calib_file):
        self.vidName = a_vidName
        # Opens the video import and sets parameters
        self.cap = Dewarp.DewarpVideoCapture(a_vidName, a_calib_file)
        self.width, self.height = (self.xCrop[1]-self.xCrop[0],self.yCrop[1]-self.yCrop[0])
        self.frameNumber = 0

    def setCrop(self, a_xCrop, a_yCrop):
        self.xCrop = a_xCrop
        self.yCrop = a_yCrop

    def plotAnts(self,image,headings):
        def draw_ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1, lineType=cv2.LINE_AA, shift=10):
            center = (int(round(center[0] * 2**shift)),int(round(center[1] * 2**shift)))
            axes = (int(round(axes[0] * 2**shift)),int(round(axes[1] * 2**shift)))
            cv2.ellipse(img, center, axes, angle,startAngle, endAngle, color, thickness, lineType, shift)

        antHeadings = [a.getPoint() for a in self.ants]
        antPredicts = [a.getPrediction() for a in self.ants]

        for heading in antPredicts:
            #print("antPredict Heading")
            #pprint(heading)
            center = (int(heading[0]),int(heading[1]))
            axes = (int(heading[4])/2,int(heading[3])/2)
            angle = int(heading[2])
            draw_ellipse(image,center,axes,angle,0,360,(0,0,255))
        for heading in antHeadings:
            #pprint(heading)
            center = (int(heading[0]),int(heading[1]))
            axes = (int(heading[4])/2,int(heading[3])/2)
            angle = int(heading[2])
            draw_ellipse(image,center,axes,angle,0,360,(255,0,0))
        for heading in headings:
            #pprint(heading)
            center = (int(heading[0]),int(heading[1]))
            axes = (int(heading[4])/2,int(heading[3])/2)
            angle = int(heading[2])
            draw_ellipse(image,center,axes,angle,0,360,(0,255,0))
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plotHeadings(self,image,headings):
        def draw_ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1, lineType=cv2.LINE_AA, shift=10):
            center = (int(round(center[0] * 2**shift)),int(round(center[1] * 2**shift)))
            axes = (int(round(axes[0] * 2**shift)),int(round(axes[1] * 2**shift)))
            cv2.ellipse(img, center, axes, angle,startAngle, endAngle, color, thickness, lineType, shift)

        for heading in headings:
            #pprint(heading)
            center = (int(heading[0]),int(heading[1]))
            axes = (int(heading[4])/2,int(heading[3])/2)
            angle = int(heading[2])
            draw_ellipse(image,center,axes,angle,0,360,(0,0,0))
            #cv2.ellipse(image, center, axes, angle, 0, 360)
            # cv2.ellipse(image,(int(heading[0]),int(heading[1])),(int(heading[3]),int(heading[4])),heading[2],0,360)
        cv2.imshow('image',image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def plotTracks(self):
        fig0 = plt.figure(figsize=(12, 10), dpi=100, facecolor='w', edgecolor='k')
        for i, a in enumerate(self.ants):
            plt.plot(a.xs, a.ys)
        plt.show()

        frames = []
        for i, frame in enumerate(self.frames):
            for ant in self.ants:
                offset = ant.time[0]
                j=i-offset
                if j >= 0:
                    print(i,offset,j)

                    pts = [(ant.xs[k],ant.ys[k]) for k in range(j)]
                    pts = np.array(pts,np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame,[pts],isClosed = False, color = (0,0,0),thickness = 2)

                    arrowstart = (int(ant.xs[j]),int(ant.ys[j]))
                    arrowend   = (int(ant.xs[j] + np.cos(ant.thetas[j])*(ant.lengths[j]/2)),\
                                    int(ant.ys[j] + np.sin(ant.thetas[j])*(ant.lengths[j]/2)))
                    cv2.arrowedLine(frame,arrowstart,arrowend,color = (0,0,0),thickness = 2,tipLength = .25)

            frames.append(frame)
            #cv2.imshow('image',frame)
            #cv2.waitKey(0)
        #Do something with frames list
        cv2.destroyAllWindows()
        """
        h,w,c = frame.shape
        print(h,w)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('project.mp4',fourcc, 30, (int(self.width),int(self.height)))

        for i in range(len(frames)):
            out.write(frames[i])
        out.release()
        """


    def trackAll(self):
        print('tracking object in all frames')
        moreFrames = True
        while moreFrames:
            moreFrames = self.processNextFrame()
            if self.frameNumber >= 5:     moreFrames = False

    def processNextFrame(self):
        print('processing frame {}'.format(self.frameNumber))
        ret, cur_frame = self.cap.read()
        if not ret: return False

        cur_frame = cur_frame[self.yCrop[0]:self.yCrop[1], self.xCrop[0]:self.xCrop[1], :]

        #===================== Threshold =======================
        # Create the basic black image
        # mask = np.zeros(cur_frame.shape, dtype = "uint8")
        # cv2.circle(mask, (900,900), 900, (255,255,255), -1)
        # cur_frame = cv2.bitwise_and(cur_frame, mask)
        # self.extractor.max = np.max(cur_frame[:,:,0])
        # self.extractor.min = np.min(cur_frame[:,:,0])
        # mask = 255*np.ones(cur_frame.shape, dtype = "uint8")
        # cv2.circle(mask, (455,455), 455, (0,0,0), -1)
        # cur_frame = cv2.bitwise_or(cur_frame, mask)
        # meas_vecs, good = self.extractor.findAnts(cur_frame, self.ants, self.frameNumber)
        # meas_vecs.append([cx, cy, theta, l, w])
        #=======================================================

        #===================== Mask RCNN =======================
        frame, temp_headings = self.extractor.findAnts(cur_frame, self.ants)

        headings = []
        for heading in temp_headings:
            if heading['type'] == "Full Ant":
                headings.append(heading['head'])
        #self.plotHeadings(cur_frame,headings)
        #=======================================================

        if self.frameNumber == 0:
            print('first iteration')
            for heading in headings:
                antNew = ant(self.dt, heading)
                self.ants.append(antNew)
        else:
            #Create an array of all ants, their headings and their distances
            distances = np.array([[j,ant.get_distance((heading[0],heading[1])),i] \
                for j, ant in enumerate(self.ants) for i, heading in enumerate(headings)\
                if ant.get_distance((heading[0],heading[1])) < 100],np.uint32)

            #Remove ant duplciates
            distances = distances[distances[:,1].argsort()]
            distances = distances[np.unique(distances[:,0],return_index=True)[1]]
            #Remove heading duplciates
            distances = distances[distances[:,1].argsort()]
            distances = distances[np.unique(distances[:,2],return_index=True)[1]]

            #print("distances: \n",distances)
            #Apply all matches
            for match in distances:
                self.ants[match[0]].add_point(headings[match[2]],self.frameNumber)
                self.ants[match[0]].update(headings[match[2]])
                #self.ants[match[0]].time[-1]=self.frameNumber

            #Find which ants have not been matched
            remaining_ants = set(range(len(self.ants))) - set(distances[:,0])
            #print("remaining_ants: \n",remaining_ants)
            #Predict new headings and update
            for a in remaining_ants:
                self.ants[a].update(self.ants[a].getPrediction())
                self.ants[a].add_point(self.ants[a].getPrediction(),self.frameNumber)


            #Find which headings have not been matched
            remaining_headings = set(range(len(headings))) - set(distances[:,2])
            #print("remaining_headings: \n",remaining_headings)
            #Create new ants
            for h in remaining_headings:
                antNew = ant(self.dt, headings[h],self.frameNumber)
                self.ants.append(antNew)

        #antHeadings = [a.getPoint() for a in self.ants]
        #self.plotAnts(cur_frame,headings)

        # print("Current Point:")
        # pprint(self.ants[0].getPoint())
        # print("Predicted Point:")
        # pprint(self.ants[0].getPrediction())
        self.frames.append(cur_frame)
        self.frameNumber += 1

        return True

    def close(self):
        cv2.destroyAllWindows()
        self.cap.cap.release()

def main():
    #Import Information
    PROJECT_DIR = os.path.join(os.getcwd(),'Tracking/Projects/sampleAnt')
    VIDEO_PATH = os.path.join(PROJECT_DIR,'AntTest.MP4')
    CALIB_DATA_PATH = os.path.join(PROJECT_DIR,'calibration_data.npz')

    tracker = antTracker()
    tracker.setCrop([1250, 2850], [200, 2500])
    tracker.setup(VIDEO_PATH, CALIB_DATA_PATH)
    tracker.trackAll()
    tracker.plotTracks()
    tracker.close()

if __name__ == "__main__":
    main()
