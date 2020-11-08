#import matplotlib
#matplotlib.use("TkAgg")
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import LineString, Point
import skimage
import statistics
import Dewarp
import os
import sys
import math
import re
import time
import logging
import tensorflow as tf
import Calibration
import MaskRCNN_TensorFlow

from pprint import pprint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.autograph.set_verbosity(0)

# Root directory of the project
ROOT_DIR = os.path.abspath(os.getcwd())

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import ant configs
from ants import ants

#This function is useful for viewing the differnt stages of the ant extraction process
def print_im(a_pic):
    #pic2Display = cv2.resize(a_pic, (800, 800))
    cv2.imshow('image', apic)
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

class Filter:
    """
    basic Kalman filter class
    """
    def __init__(self, a_initialState, a_initialStateCov,
                 a_measCov, a_modelCov, a_model, a_state2Meas):
        """initialization function"""
        self.xk = a_initialState
        self.Pk = a_initialStateCov
        self.Q = a_modelCov
        self.R = a_measCov
        self.F = a_model
        self.H = a_state2Meas

    def update_extra(self):
        pass

    def update(self, a_meas):
        """This function updates the state of the ant given a new measurement a_meas
        Nothing should need changed here
        """

        # get predicted state and covar
        xp, Pp = self.predict(self.xk, self.Pk)

        # get predicted measurement and measurement Covariance
        zp, K = self.predictedMeas(xp, Pp)

        # update the state vector
        zdiff = a_meas - zp

        # correct thing if need. This is useful for things like angles that should be between 0 and 360 deg.
        zdiff = self.correctMeasSubtract(zdiff, zp, a_meas)

        # update the state vector
        self.xk = xp + np.dot(K, zdiff)

        # update the state covar
        self.Pk = Pp - np.dot(K, np.dot(self.H, Pp))

        # correct thing if need. This is useful for things like angles that should be between 0 and 360 deg.
        self.xk, self.Pk = self.CorrectUpdate(self.xk, self.Pk)
        self.update_extra()

        return self.xk, self.Pk

    def predictedMeas(self, a_xp, a_Pp):
        """This function tranforms the state vector and covariance matrix into measurement space.
        Nothing should need changed here provided you have your matrices H,
        and R set correctly"""
        zp = np.dot(self.H, a_xp)
        S = np.dot(self.H, np.dot(a_Pp, self.H.T)) + self.R
        K = np.dot(a_Pp, np.dot(self.H.T, np.linalg.inv(S)))

        zp, S, K = self.predictedMeasCorrection(zp, S, K)
        return zp, K

    def predict(self, a_state, a_stateCov):
        """ prediction function this gives the predicted state vector and
        covariance matrix given the ones from the previous time step.
        Nothing should need changed here provided you have your matrices F,
        and Q set correctly
        """
        xp = np.dot(self.F, a_state)  # predicted state vector
        Pp = np.dot(self.F, np.dot(a_stateCov, self.F.T)) + self.Q  # predicted state covariance matrix

        # correct thing if need. This is useful for things like angles that should be between 0 and 360 deg.
        xp, Pp = self.predictionCorrection(xp, Pp)

        return xp, Pp

    def updateH(self, a_H):
        """This function allows you to update the H matrix incase it changes with time."""
        self.H = a_H

    def updateF(self, a_F):
        """This function allows you to update the F matrix incase it changes with time."""
        self.F = a_F

    def updateQ(self, a_Q):
        """This function allows you to update the Q matrix incase it changes with time."""
        self.Q = a_Q

    def updateR(self, a_R):
        """This function allows you to update the R matrix incase it changes with time."""
        self.R = a_R

    def predictionCorrection(self, a_xp, a_Pp):
        """ add your correction here if one is needed. """
        return a_xp, a_Pp

    def predictedMeasCorrection(self, a_zp, a_S, a_K):
        """ add your correction here if one is needed. """
        return (a_zp, a_S, a_K)

    def correctMeasSubtract(self, a_zdiff, a_zp, a_zk):
        """ add your correction here if one is needed. """

        return a_zdiff

    def CorrectUpdate(self, a_xk, a_Pk):
        """ add your correction here if one is needed. """

        return a_xk, a_Pk


class ant(Filter):
    def __init__(self, dt, a_initialState):

        #self.xk = a_initialState
        self.xk = self.toPoint(a_initialState)

        self.Pk = np.array([[20000, 0, 0, 0, 0],
                            [0, 20000, 0, 0, 0],
                            [0, 0, 300, 0, 0],
                            [0, 0, 0, 1000, 0],
                            [0, 0, 0, 0, 1000]])

        self.R = np.array([[100, 0, 0, 0, 0],
                           [0, 100, 0, 0, 0],
                           [0, 0, 60, 10, 10],
                           [0, 0, 10, 80, 0],
                           [0, 0, 10, 0, 80]])

        self.F = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
                           [0, 0, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 1]])

        self.Q = np.array([[8000, 200, 0, 0, 0],
                           [200, 8000, 0, 0, 0],
                           [0, 0, 80, 0, 0],
                           [0, 0, 0, 4, 0],
                           [0, 0, 0, 0, 4]])

        self.H = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0],
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
        self.add_point(self.xk, 0)

    def toHead(self, a_temp):
        if type(a_temp) is dict:
            return a_temp
        else:
            point = a_temp
            head = {
                "a":point[3]*point[4],
                "x":point[0],
                "y":point[1],
                "w":point[4],
                "l":point[3],
                "t":point[2]
            }
            return head

    def toPoint(self, a_temp):
        if type(a_temp) is dict:
            head = a_temp
            point = [head['x'],head['y'],head['t'],head['l'],head['w']]
            return(point)
        else:
            return(a_temp)

    def update_extra(self):
        self.add_point(self.xk, 0)

    def update(self, a_pointOhead):
        super().update(self.toPoint(a_pointOhead))

    def get_distance(self, a_point):
        return np.sqrt((self.xs[-1] - a_point[0]) ** 2 + (self.ys[-1] - a_point[1]) ** 2)

    def add_point(self, a_pointOhead, a_time):
        point = self.toHead(a_pointOhead)
        #self.xs.append(a_point[0])
        #self.ys.append(a_point[1])
        self.xs.append      (point['x'])
        self.ys.append      (point['y'])
        self.lengths.append (point['l'])
        self.widths.append  (point['w'])
        self.thetas.append  (point['t'])
        self.areas.append   (point['a'])

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

        if np.abs(a_zdiff[2]) > 90:
            a_zdiff[2] = a_zdiff[2] - np.sign(a_zdiff[2]) * 180

        return a_zdiff

    def CorrectUpdate(self, a_xk, a_Pk):
        """ add your corretion here if one is needed. """
        self.predictionCorrection(a_xk, a_Pk)
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
        self.set_log_dir()#self.model_dir)
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
class extractorConfig(ants.AntConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.3

class extractorMaskRCNN(MaskRCNN_TensorFlow.MaskRCNN_TensorFlow):
    def __init__(self,a_model_dir = None,a_weights_path = None):
        # Directory to save logs and trained model
        if a_model_dir: model_dir = a_model_dir
        else:           model_dir = os.path.join(ROOT_DIR, "logs")

        if a_weights_path:  WEIGHTS_PATH = a_weights_path
        else:               WEIGHTS_PATH = os.path.join(ROOT_DIR,'models/TRAINEDFULLANTS824.h5')

        self.config = extractorConfig()
        #self.config.display()

        # Create model object in inference mode.
        self.model = detectorMaskRCNN(mode="inference", model_dir=model_dir, config=self.config)

        #model.load_weights(MODEL_PATH, by_name=True)
        self.model.load_weights(WEIGHTS_PATH, by_name=True)

    def findAnts(self, image, ants):
        #try: image = skimage.io.imread(a_file_name)
        #except e: return None, None

        results = self.model.detect([image], verbose=0)
        r = results[0]

        # convert image to open_cv so it works with what we have already done
        # cv_image = skimage.img_as_ubyte(image)
        # cv_image = np.ubyte(image)

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
            #For now, check if the class_id is full ant
            if class_ids[r['class_ids'][i]] == 'Full Ant':
                mask = r['masks'][:,:,i]
                contours, _ = cv2.findContours(mask.astype('uint8'), mode = cv2.RETR_LIST, method = cv2.CHAIN_APPROX_SIMPLE)
                cnt = contours[0]
                ellipse = cv2.fitEllipse(cnt)
                antLocs.append({
                    "a":cv2.contourArea(cnt),
                    "x":ellipse[0][0],
                    "y":ellipse[0][1],
                    "w":ellipse[1][0],
                    "l":ellipse[1][1],
                    "t":ellipse[2]
                })

                # areas.append(cv2.contourArea(cnt))
                # cx.append(ellipse[0][0])
                # cy.append(ellipse[0][1])
                # ws.append(ellipse[1][0])
                # ls.append(ellipse[1][1])
                # thetas.append(ellipse[2])
        ##########################

        """
        Output:

        Headings, a list of dicts, each of the form:
        {
            "a":area,
            "x":center x coord,
            "y":center y coord,
            "w":width,
            "l":length,
            "t":theta
        }
        """

        vis = None #
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],class_ids, r['scores'], title="Predictions")

        #should return a list of the form: [cx, cy, theta, l, w]
        return vis, antLocs #[cx,cy,thetas,ls,ws] #self.reformat(r['rois'],r['masks'], r['scores'], r['class_ids']), cv_image[:,:,::-1]

class antTracker:
    def __init__(self):
        self.dt = 1
        self.extractortype = "MaskRCNN"
        if self.extractortype == "Threshhold":
            self.extractor = extractorTH
        else:
            self.extractor = extractorMaskRCNN()
        self.ants = []
        self.frames = []

        self.colors = ['r', 'g', 'b', 'k', 'm']

    def setup(self, a_vidName, a_calib_file):
        self.vidName = a_vidName
        # Opens the video import and sets parameters
        self.cap = Dewarp.DewarpVideoCapture(a_vidName, a_calib_file)
        width, height = (self.xCrop[1]-self.xCrop[0],self.yCrop[1]-self.yCrop[0])
        self.frameNumber = 0

    def setCrop(self, a_xCrop, a_yCrop):
        self.xCrop = a_xCrop
        self.yCrop = a_yCrop

    def plotTracks(self, overFrames = False):
        if not overFrames:
            fig0 = plt.figure(figsize=(12, 10), dpi=100, facecolor='w', edgecolor='k')
            for i, a in enumerate(self.ants):
                plt.plot(a.xs, a.ys )#, self.colors[i])
            plt.show()

    def trackAll(self):
        print('tracking object in all frames')
        moreFrames = True
        while moreFrames:
            #print(self.frameNumber)
            moreFrames = self.processNextFrame()
            if self.frameNumber >= 100:
                moreFrames = False

    def processNextFrame(self):
        print('processing frame {}'.format(self.frameNumber))

        ret, cur_frame = self.cap.read()
        if not ret:
            return False #True

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
        frame, headings = self.extractor.findAnts(cur_frame, self.ants)

        """headings is of the form:
        {
            "a":area,
            "x":center x coord,
            "y":center y coord,
            "w":width,
            "l":length,
            "t":theta
        }
        """
        #=======================================================

        if self.frameNumber == 0:
            print('first iteration')
            for heading in headings:
                antNew = ant(self.dt, heading)
                self.ants.append(antNew)
        else:
            print('frame number: ',self.frameNumber,' ants found: ',len(headings))
            pprint(self.ants[0])
            #pprint(headings[0])

            used = set()
            remaining = set(range(len(self.ants)))
            matchInf = []

            #if len(self.ants) != len(meas_vecs):
            #    print(len(ants), len(meas_vecs), "  There was a problem")

            while len(remaining) > 0:
                dists = []
                i = list(remaining)[0] #ant ind
                print("remaining:")
                pprint(remaining)
                for j, heading in enumerate(headings):
                    #print("Comparing heading : ",j," to ant : ",i)
                    #pprint(heading)
                    #pprint([self.ants[i].xs[-1],self.ants[i].ys[-1],self.ants[i].thetas[-1]])

                    dist = self.ants[i].get_distance((heading['x'],heading['y']))
                    dists.append(dist)

                matchedInd = np.argmin(dists)
                #print("distances: ",dists)
                #print("matchedInd: ",matchedInd)
                #print("matched index: ",matchedInd)

                matchInf.append([i, np.min(dists), matchedInd])
                #matchInf[matchedInd] = [i,np.min(dists)]

                print("Match Inf: ")
                pprint(matchInf)

                remaining.remove(i)

                """
                sortingStuff = True
                while sortingStuff:
                    matchedInd = np.argmin(dists) #contour ind
                    #check if it has already be assigned
                    if matchedInd in used:
                        #is the previous assignment better?
                        #if np.min(dists) >= matchInf[matchedInd][1]:
                        if np.min(dists) >= matchInf[np.where(matchInf)][2]
                            dists[matchedInd] = 9999999  # make it large so we dont find it again
                        else: #if not lets change it
                            #add back the old index
                            remaining.add(matchInf[matchedInd][0])
                            #over write the match info
                            matchInf[matchedInd] = [i, np.min(dists)]
                            remaining.remove(i)
                            sortingStuff = False
                    else:
                        used.add(matchedInd)
                        #matchInf[matchedInd] = [i,np.min(dists)]
                        matchInf.append([i, np.min(dists), matchedInd])
                        remaining.remove(i)
                        sortingStuff = False
                """

            for match in matchInf:#.items():
                    #meas_vect, __ = self.ants[match[0]].predictionCorrection(np.array(meas_vects[match[2]]), None)
                    #self.ants[match[0]].update(meas_vect)
                    self.ants[match[0]].update(headings[match[2]])
                    self.ants[match[0]].time[-1]=self.frameNumber

        self.frameNumber += 1

        #self.frames.append(image)

        return True

    def close(self):
        cv2.destroyAllWindows()
        self.cap.cap.release()

def main():
    #Import Information
    PROJECT_DIR = os.path.join(os.getcwd(),'Projects/sampleAnt')
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
