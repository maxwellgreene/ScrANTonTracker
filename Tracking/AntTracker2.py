#import matplotlib
#matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import LineString, Point
import statistics

import Dewarp



#This function is useful for viewing the differnt statges of the ant extraction process
def print_im(a_pic):
    pic2Dispaly = cv2.resize(a_pic, (800, 800))
    cv2.imshow('image', pic2Dispaly)
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
        self.xk = a_initialState

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

        self.state_vars = []
        self.state_covars = []
        self.meas_vars = []
        self.xs = []
        self.ys = []
        self.time = []
        self.add_point(a_initialState, 0)

    def update_extra(self):
        self.add_point(self.xk, 0)

    def get_distance(self, a_point):
        return np.sqrt((self.xs[-1] - a_point[0]) ** 2 + (self.ys[-1] - a_point[1]) ** 2)

    def add_point(self, a_point, a_time):
        self.xs.append(a_point[0])
        self.ys.append(a_point[1])
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


class extractorTH:
    def __init__(self):
        self.num_ants = None
        self.antsWithCont = []
        self.antsWithoutCont = []
        self.thresh = 145
        self.thresh_shift = 0
        self.valid_contours = []
        self.m = 0
        self.M = 255

    def set_ant_num(self, a_num_ants):
        """ Used to manualy set the nuymber of ants.
        Latter we should add in the abilty to do this manualy on the first frame
        if it has not been set manualy"""
        # if isinstance(x, int):
        self.num_ants = a_num_ants

    def get_num_ants(self):
        num_ants = int(input("Howmany ants are there? \n"))
        self.set_ant_num(num_ants)

    def get_len_width(self, a_box):
        p1 = a_box[0]

        distances = []
        for p in a_box[1:]:
            d = np.sqrt((p[0] - p1[0]) ** 2 + (p[1] - p1[1]) ** 2)
            distances.append(d)

        distances.sort()
        return distances[:-1]

    def get_len_width_theta(self, a_box):

        p1 = a_box[0]

        distances = []
        thetas = []

        for p in a_box[1:]:
            dx = p[0] - p1[0]
            dy = p[1] - p1[1]
            if dx == 0:
               thetas.append(90.0)
            else:
                thetas.append(np.arctan(dy / dx) * 180 / 3.14159)

            distances.append(np.sqrt(dx ** 2 + dy ** 2))

        out = sorted(zip(distances, thetas))
        distances, thetas = zip(*out)
        return distances[0], distances[1], thetas[1]

    def get_corners(self, length, width, theta, center_x, center_y, scale):
        xt = scale * length / 2 * np.cos(theta * 3.14159 / 180)
        yt = scale * length / 2 * np.sin(theta * 3.14159 / 180)

        xs = scale * width / 2 * np.cos((theta + 90) * 3.14159 / 180)
        ys = scale * width / 2 * np.sin((theta + 90) * 3.14159 / 180)

        p1 = [int(center_x + xt + xs), int(center_y + yt + ys)]
        p2 = [int(center_x + xt - xs), int(center_y + yt - ys)]
        p3 = [int(center_x - xt - xs), int(center_y - yt - ys)]
        p4 = [int(center_x - xt + xs), int(center_y - yt + ys)]

        return np.array([p1, p2, p3, p4])

    def make_markers(self, ants, a_frame):
        x, y = a_frame.shape  # I did
        markers = np.zeros((x, y), np.uint8)

        polys = []
        for ant in ants:
            tempMask = np.zeros((x, y), np.uint8)
            tempMask[a_frame > 0] = 255

            xp, Pp = ant.predict(ant.xk, ant.Pk)
            length = xp[3]
            width = xp[4]
            theta = xp[2]
            center_x = xp[0]
            center_y = xp[1]
            box = self.get_corners(length, width, theta, center_x, center_y, .25)
            poly = Polygon(box)

            antMask = mask_for_polygons([poly], a_frame.shape)
            tempMask[antMask == 0] = 0

            if np.amax(tempMask) > 0:
                markers[tempMask > 0] = 255
            else:
                markers[antMask > 0] = 255

            polys.append(poly)


        # make the unknown regions
        unknown = np.zeros((x, y), np.uint8)
        unknown[a_frame > 20] = 255
        mask = mask_for_polygons(polys, a_frame.shape)
        unknown[mask > 0] = 255
        # mask out the ant centers
        unknown[markers > 0] = 0

        # make the final marker image
        # Marker labelling
        ret, markers = cv2.connectedComponents(markers)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        return markers

    

    def draw_boarders(self, ants, a_frame):
        for ant in ants:
            xp, Pp = ant.predict(ant.xk, ant.Pk)
            length = xp[3]
            width = xp[4]
            theta = xp[2]
            center_x = xp[0]
            center_y = xp[1]
            box = self.get_corners(length, width, theta, center_x, center_y, 1.)

            cv2.drawContours(a_frame, [box], 0, (0, 0, 0), 2)

        return a_frame

    def makeBlobs(self, a_img):
        gray = a_img[:,:,0] #cv2.cvtColor(a_img, cv2.COLOR_BGR2GRAY)
        #clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
        print_im(gray)
        gray = (255/(self.M-self.m))*(gray-self.m)
        gray[gray>255] = 255
        print('rescaled', self.m, self.M)
        print(np.min(gray), np.max(gray))
        g = gray.astype("uint8")
        gray = g
        print_im(gray)

        # blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        ret, thresh = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY_INV)
        
        """
        thresh2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,55,50)
        thresh3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,55,50)
        print_im(thresh)
        print_im(thresh2)
        print_im(thresh3)
        """
        print("thresh")
        print_im(thresh)
        closedIM = self.close(thresh  , 5)
        """

        # Perform the operation
        output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]
        for ant in ants:
            tempMask = np.zeros((x, y), np.uint8)
            tempMask[a_frame > 0] = 255

            xp, Pp = ant.predict(ant.xk, ant.Pk)
            length = xp[3]
            width = xp[4]
            theta = xp[2]
            center_x = xp[0]
            center_y = xp[1]
            box = self.get_corners(length, width, theta, center_x, center_y, .25)
            poly = Polygon(box)

        """
        #print_im(closedIM)
        return closedIM


    def filterContours(self, a_contours):
        areas = []
        lens = []

        # get info on all the connected components
        for i, cnt in enumerate(a_contours):
            area = cv2.contourArea(cnt)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            w, l, theta = self.get_len_width_theta(box)

            lens.append(l)
            areas.append(area)

        if len(areas) ==0:
            return []
        median_len = statistics.median(lens)
        median_area = statistics.median(areas)

        contours_out = []

        for a, l, cnt in zip(areas, lens, a_contours):
            if 100 * a / median_area > 60 and 100 * l / median_len > 60:  # apply median filter
                contours_out.append(cnt)

        return contours_out

    def resetForRun(self):
        self.valid_contours = []
        self.antsWithCont = []
        self.antsWithoutCont = []

    def findAnts(self, a_frame, a_ants, a_frame_num):
        allFound = False

        if a_frame_num == 0:
            print_im(a_frame[:,:,0])
            #print_im(a_frame[:,:,1])
            #print_im(a_frame[:,:,2])

        # make sure we know how many ants there should be.
        if self.num_ants is None:
            self.get_num_ants()

        # reset the containers
        self.resetForRun()

        count = 0
        ants2find = set(range(len(a_ants)))
        
        while not allFound:
            count += 1
            #print("the threshold is ", self.thresh)
            # Get the connected componenst of the threshold image
            closedIM = self.makeBlobs(a_frame)
            print_im(closedIM)
            # get the contours of the threhold blobs
            contours, hierarchy = cv2.findContours(closedIM, 1, 2)[-2:]
            #cv2.connectedComponentsWithStats(closedIM, 8, cv2.CV_32S)

            #if len(contours) > self.num_ants:
            contours = self.filterContours(contours)
            print("we found ", len(contours), " ants")

            # check how we did and adjust things if need
            if a_frame_num > 0:
                #if len(contours) == self.num_ants:  # everything is good on the first frame and we can proceed.
                self.valid_contours = contours
                allFound = True
                continue
                
                markers = self.make_markers(a_ants, closedIM)
                # plt.figure(figsize=(12, 4), dpi= 100, facecolor='w', edgecolor='k')
                # plt.imshow(markers)
                # plt.show()
                markers = cv2.watershed(a_frame, markers)
                #mset = np.unique(markers)
                #print(len(mset), " and the set is ", mset)
                
                #plt.figure(figsize=(12, 4), dpi= 100, facecolor='w', edgecolor='k')
                #plt.imshow(markers)
                #plt.show()
                allFound = True

                for i in np.arange(2, np.amax(markers) + 1):
                    #print(i)
                    tmask = np.zeros(markers.shape, np.uint8)
                    tmask[markers == i] = 255;

                    conts, hierarchy = cv2.findContours(tmask, 1, 2)[-2:]
                    self.valid_contours.append(conts[0])
                #print("len valid_conts", len(self.valid_contours))
            # if we are a on the first frame make sure we have the write number of ants
            if a_frame_num == 0 and len(contours) < self.num_ants:  # threshold is too low
                self.thresh += 1
                print("changeTheshUp")
            elif a_frame_num == 0 and len(contours) > self.num_ants:  # threshold is too high
                print("changeThesh down")
                self.thresh -= 1
            elif a_frame_num == 0:  # everything is good on the first frame and we can proceed.
                self.valid_contours = contours
                allFound = True

            # if this fails then the frame is bad and we need to proceed without it
            if self.thresh <= 30 or self.thresh >= 230 or count >= 200:
                self.thresh = 90
                self.thresh_shift = 0
                return [], False

        # print(len(contours))
        contours = self.valid_contours

        areas = []
        boxes = []
        centers = []
        lens = []
        widths = []
        thetas = []
        meas_vecs = []

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            w, l, theta = self.get_len_width_theta(box)

            # compute the center of the contour
            M = cv2.moments(cnt)
            if M["m00"] < .0001: continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append([cx, cy])

            lens.append(l)
            widths.append(w)
            thetas.append(theta)
            areas.append(area)
            boxes.append(box)

            # print('out', w,l,theta)
            meas_vecs.append([cx, cy, theta, l, w])
            cv2.drawContours(a_frame, [box], 0, (0, 0, 255), 2)
            # print_im(a_frame)



        # print_im(a_frame)
        # print(len(meas_vecs))
        if len(meas_vecs) == self.num_ants:
            # print('good',len(keep_centers))
            return meas_vecs, True
        else:
            # print('bd',len(keep_centers))
            return meas_vecs, True
            #return [], False

    def close(self, a_frame, a_size):
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(a_frame, kernel, iterations=2)
        #print_im(erosion)
        kernel = np.ones((a_size, a_size), np.uint8)
        #dilation = cv2.dilate(erosion, kernel, iterations=3)
        #print_im(dilation)
        return erosion




class antTracker:
    def __init__(self):
        self.dt = 1
        self.extractor = extractorTH()
        self.ants = []
        self.colors = ['r', 'g', 'b', 'k', 'm']

    def setup(self, a_vidName, a_calib_file):
        self.vidName = a_vidName
        # Opens the video import and sets parameters
        self.cap = Dewarp.DewarpVideoCapture(a_vidName, a_calib_file)
        self.frameNumber = 0

    def setCrop(self, a_xCrop, a_yCrop):
        self.xCrop = a_xCrop
        self.yCrop = a_yCrop

    def plotTracks(self):
        fig0 = plt.figure(figsize=(12, 10), dpi=100, facecolor='w', edgecolor='k')

        for i, a in enumerate(self.ants):
            plt.plot(a.xs, a.ys)  # , self.colors[i])
        plt.show()

    def trackAll(self):
        print('tracking object in all frames')
        moreFrames = True
        while moreFrames:
            print(self.frameNumber)
            moreFrames = self.processNextFrame()
            #if self.frameNumber >= 10:
            #    moreFrames = False


    def processNextFrame(self):
        # print('processing frame {}'.format(self.frameNumber))

        ret, cur_frame = self.cap.read()
        print('read frame')
        if (type(cur_frame) == type(None)):
            
            return True

        cur_frame = cur_frame[self.yCrop[0]:self.yCrop[1], self.xCrop[0]:self.xCrop[1], :]


        # Create the basic black image 
        #mask = np.zeros(cur_frame.shape, dtype = "uint8")
        #cv2.circle(mask, (900,900), 900, (255,255,255), -1)
        #cur_frame = cv2.bitwise_and(cur_frame, mask)
        self.extractor.M = np.max(cur_frame[:,:,0])
        self.extractor.m = np.min(cur_frame[:,:,0])                           
        mask = 255*np.ones(cur_frame.shape, dtype = "uint8")
        cv2.circle(mask, (455,455), 455, (0,0,0), -1)
        cur_frame = cv2.bitwise_or(cur_frame, mask)        

        meas_vecs, good = self.extractor.findAnts(cur_frame, self.ants, self.frameNumber)
        if not good:
            print('not good')
            self.frameNumber+=1
            return True

        if self.frameNumber == 0:
            #print('first iteration')
            for meas_vec in meas_vecs:
                antNew = ant(self.dt, np.array(meas_vec))
                self.ants.append(antNew)
        else:
            #print('second iteration')
            #print(len(meas_vecs))
            used = set()
            remaining = set(range(len(self.ants)))
            #matchInf = {}
            matchInf = []

            #if len(self.ants) != len(meas_vecs):
            #    print(len(ants), len(meas_vecs), "  THere was a problem")
            #    print(bob)
            while len(remaining) > 0:
                dists = []
                i = list(remaining)[0] #ant ind
                for meas_vec in meas_vecs:
                    # print(meas_vec)
                    dist = self.ants[i].get_distance(np.array(meas_vec))
                    dists.append(dist)
                    
                matchedInd = np.argmin(dists)
                matchInf.append( [i, np.min(dists), matchInd])
                """
                sortingStuff = True
                while sortingStuff:
                    matchedInd = np.argmin(dists) #contour ind
                    #check if it has already be assigned
                    if matchedInd in used:
                        #is the previouse assignment better?
                        if np.min(dists) >= matchInf[matchedInd][1]:
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
                        matchInf[matchedInd] = [i, np.min(dists)]
                        remaining.remove(i)
                        sortingStuff = False
                """
            for  match in matchInf:#.items():
                    meas_vec, __ = self.ants[match[0]].predictionCorrection(np.array(meas_vecs[match[2]]), None)
                    self.ants[match[0]].update(meas_vec)
                    self.ants[match[0]].time[-1]=self.frameNumber
                    
        self.frameNumber += 1

        frame = self.extractor.draw_boarders(self.ants, cur_frame)
        print_im(frame)

        return True

    def close(self):
        cv2.destroyAllWindows()
        self.cap.cap.release()



def main():
    filename = "Cliped.MP4" #"GP010871.MP4"
    calib_file = 'calibration_data.npz'

    tracker = antTracker()
    tracker.setup(filename, calib_file)
    tracker.setCrop([650, 1550], [110, 1020])
    tracker.trackAll()
    tracker.plotTracks()
    tracker.close()


if __name__ == "__main__":
    main()
