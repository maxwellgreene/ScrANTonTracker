import matplotlib.pyplot as plt
import GoPro_calib as GoProCal
import cv2, sys
import numpy as np
import time


def makeUndestortMeta():

    #Import Information
    filename = 'Videos/pattern2.MP4'
    #Input the number of board images to use for calibration (recommended: ~20)
    n_boards = 30
    #Input the number of squares on the board (width and height)
    board_w = 9
    board_h = 6
    #Board dimensions (typically in cm)
    board_dim = 25
    #Image resolution
    image_size = (3840,2160)#(1920, 1080)

    #Crop mask
    # A value of 0 will crop out all the black pixels.  This will result in a loss of some actual pixels.
    # A value of 1 will leave in all the pixels.  This maybe useful if there is some important information
    # at the corners.  Ideally, you will have to tweak this to see what works for you.
    crop = 1

    sc = .25 #rescaling factor so that images fit on the screen


    print("Starting camera calibration....")
    print("Step 1: Image Collection")
    print("We will playback the calibration video.  Press the spacebar to save")
    print("calibration images.")
    print(" ")
    print('We will collect ' + str(n_boards) + ' calibration images.')

    #GoProCal.ImageCollect(filename, n_boards)

    print(' ')
    print('All the calibration images are collected.')
    print('------------------------------------------------------------------------')
    print('Step 2: Calibration')
    print('We will analyze the images take and calibrate the camera.')
    print('Press the esc button to close the image windows as they appear.')
    print(' ')

    GoProCal.ImageProcessing(n_boards, board_w, board_h, board_dim)


class DewarpVideoCapture:
    def __init__(self, a_filename,
                 a_calibration_file='calibration_data.npz'):

        # call the constructor form cv2.videoCapture
        self.cap = cv2.VideoCapture(a_filename)

        # set up the dewarping stuff
        npz_calib_file = np.load(a_calibration_file)
        distCoeff = npz_calib_file['distCoeff']
        intrinsic_matrix = npz_calib_file['intrinsic_matrix']
        npz_calib_file.close()

        # Crop mask
        # A value of 0 will crop out all the black pixels.  This will result in a loss of some actual pixels.
        # A value of 1 will leave in all the pixels.  This maybe useful if there is some important information
        # at the corners.  Ideally, you will have to tweak this to see what works for you.
        crop = 0

        if self.cap.isOpened():
            width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.size = (int(width), int(height))

            newMat, ROI = cv2.getOptimalNewCameraMatrix(intrinsic_matrix, distCoeff, self.size,
                                                        alpha=crop, centerPrincipalPoint=1)

            self.mapx, self.mapy = cv2.initUndistortRectifyMap(intrinsic_matrix, distCoeff,
                                                               None, newMat, self.size,
                                                               m1type=cv2.CV_32FC1)

    def read(self):
        success, image = self.cap.read()
        return success, image;
        """if image is None:
            return success, image

        dewarped_image = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)
        return success, dewarped_image
        """

