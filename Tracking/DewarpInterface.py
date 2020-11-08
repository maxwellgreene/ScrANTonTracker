import sys
#import cv2
import numpy as np
import os
import Calibration

WORKING_DIR = os.getcwd()
VIDEO_DIR = os.path.join(WORKING_DIR,'Projects')

mode = 'none'

if __name__ == '__main__':

    if len(sys.argv) > 1:
        if str(sys.argv[1]) == 'help':
            print('')
            print('============================================================================================')
            print('=========================== Printing a list of possible modes... ===========================')
            print('============================================================================================ \n')

            print('--------------------------------------- MODE #1 -------------------------------------------- \n')
            print('Save a calibration file to a given name inside the project folder')
            print('<py file path> \"saveCalib\" <projectName> <videoName> <writeNameData = calibration_data> \n')

            print('--------------------------------------- MODE #2 -------------------------------------------- \n')
            print('Dewarp a given video using a given dewarp calibration file')
            print('Note: file will be saved as \"Dewarped<videoName>\.MP4" ')
            print('<py file path> \"dewarpVideo\" <projectName> <videoName> <dataFile = calibration_data> \n')
            print('-------------------------------------------------------------------------------------------- \n')
            exit()

        if str(sys.argv[1].lower()) == 'savecalib':
            mode = sys.argv[1]              #Argument 1  ...  mode
            projectName = sys.argv[2]       #Argument 2  ...  project name
            readNameVideo = sys.argv[3]     #Argument 3  ...  video name
            if len(sys.argv) >= 5:
                writeNameData = sys.argv[4] #Argument 4  ...  calibration data write name
            else:
                writeNameData = 'calibration_data'
            if len(sys.argv) > 5 :
                print('Too many arguments... Exiting.'); exit()

            readPathVideo = os.path.join(VIDEO_DIR,projectName,readNameVideo+'.MP4')
            writePathData = os.path.join(VIDEO_DIR,projectName,(writeNameData+'.npz'))

        if str(sys.argv[1].lower()) == 'dewarpvideo':
            mode = str(sys.argv[1])
            projectName = str(sys.argv[2])
            readNameVideo = str(sys.argv[3])
            #print("readNameVideo: ",readNameVideo)
            if len(sys.argv) >= 5:
                readNameData = str(sys.argv[4])
            else:
                readNameData = 'calibration_data'
            if len(sys.argv) > 5 :
                print('Too many arguments... Exiting.'); exit()

            readPathVideo = os.path.join(VIDEO_DIR,projectName,readNameVideo)
            readPathData = os.path.join(VIDEO_DIR,projectName,(readNameData + '.npz'))
            writeNameVideo = ('Dewarped' + readNameVideo)
            writePathVideo = os.path.join(VIDEO_DIR,projectName,('Dewarped' + readNameVideo))
            try:
                open(readPathData)
            except IOError :
                print('====================================================================================')
                print('================   Could not find data file at given location   ====================')
                print('==========   Have you saved the calibration file for this project yet?   ===========')
                print('====================================================================================')
                exit()

        if str(sys.argv[1]) == 'nextMode':
            mode = str(sys.argv[1]).lower()
            #whatever else...


    if mode == 'none':
        print(' ')
        print('Hello, fellow idiot. I\'m not sure what you\'re trying to do. Exiting.'); exit()
        print(' ')

    if mode == 'savecalib':
        print(' ')
        print(('Generating calibration file for '+readNameVideo+' and saving as '+writeNameData))
        print(' ')

        Calibration.ImageCollect(readPathVideo)
        Calibration.ImageProcessing(writePathData)

    if mode == 'dewarpvideo':
        print(' ')
        print(('Dewarping movie file '+readNameVideo+' and saving as '+writeNameVideo))
        print(' ')

        Calibration.DewarpMovie(readPathVideo, writePathVideo, readPathData)
