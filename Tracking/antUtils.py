import os
import sys
import random
import skimage.io
from skimage import img_as_ubyte
from skimage import img_as_float
import colorsys
import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath(os.getcwd())

# Import ant configs
import Ant

def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors

class InferenceConfig(Ant.AntConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.3


class MaskRCNN_TensorFlow:
    def __init__(self):
        # Directory to save logs and trained model
        MODEL_DIR = os.path.join(ROOT_DIR, "logs")

        # Local path to trained weights file
        WEIGHTS_PATH = os.path.join(ROOT_DIR,'models/TRAINEDFULLANTS.h5')

        self.config = InferenceConfig()
        #self.config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)

        #model.load_weights(MODEL_PATH, by_name=True)
        self.model.load_weights(WEIGHTS_PATH, by_name=True)

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        self.class_names = ['BG','Full Ant', 'Head','Thorax', 'Abdomen']
        self.labels = {}
        for i, name, in enumerate(self.class_names):
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

    def reformat(self, rois, masks, scores, class_ids):
        boxes_out = []
        i=0
        for roi, score, class_id in zip(rois, scores, class_ids):
            #print(box)
            (startX, startY, endX, endY) = roi.astype("int")
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

        # todo remove writing here as it should happen later.
        # cv2.imwrite(a_out_file, clone)

    def write_on_image(self, image, text_str, location, size):
        cv2.putText(image,
                    text_str,
                    location,
                    fontFace=cv2.FONT_HERSHEY_DUPLEX,
                    fontScale=size,
                    color=(0, 0, 255))



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
