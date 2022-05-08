import csv
import multiprocessing

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from PIL import Image
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pickle5 as pickle

with open('SVC_model_Eye_Blinking_Trained_Model_Eyeblink8', 'rb') as handle:
    SVC_model = pickle.load(handle)

# eye parameter is just an array containing all the (x, y) coordinates of all the eye
def eyeAspectRatioFunction(eye):
  #  print("eye size", eye.size)
    # get the distance between the vertical eye landmarks - there are two vertical eye landmarks for a given eye

    verticalA = dist.euclidean(eye[1], eye[5])
    verticalB = dist.euclidean(eye[2], eye[4])

    # get the distance between the horizontal eye landmarks
    horizontalA = dist.euclidean(eye[0], eye[3])

    # find the eye aspect ratio
    eyeAspectRatio = (verticalA + verticalB) / (2.0 * horizontalA)


    return eyeAspectRatio


# command line argument
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-p", "--shape-predictor", required=True,
                             help="path to facial landmark predictor")
argument_parser.add_argument("-v", "--video", type=str, default="",
                             help="path to input video file")
args = vars(argument_parser.parse_args())

detector = dlib.get_frontal_face_detector() # used to find and locate face in each frame
predictor = dlib.shape_predictor(args["shape_predictor"]) # used to extract faical feature of a given face

# variables
BLINK_EYE_FRAMES_NEEDED = 3
OPEN_EYE_FRAMES_NEEDED = 10
SquintTestCounterThreshold = 0
SquintTestCounter = 0
hasFrameBlinkHappened = False
testForUserSquinting = False
BlinkFrame_COUNTER = 0
OpenFrame_COUNTER = 0
Previous_OpenFrame_COUNTER = 0
Previous_Blink_COUNTER = 0
TOTAL = 0
FrameCounter = 0

# get the facial landmarks points
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

#######################
# start the video stream thread
#########################
vs = FileVideoStream(args["video"]).start()
fileStream = True

time.sleep(1.0)
totalBlinkFrames= 0
potenialBlinkFrame = []
# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not vs.more():
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()

    frame = imutils.resize(frame, width=800)
    #  img = Image.open(frame).convert('LA')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[0:6]
        rightEye = shape[6:12]
        leftEAR = eyeAspectRatioFunction(leftEye)
        rightEAR = eyeAspectRatioFunction(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        print("FrameCounter = ", FrameCounter, "ear = ", ear)

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

        # for number 5
        earPredict = np.array(ear)
        earPredict = earPredict.ravel()

        if earPredict > 0.20:
            classifier = "Open"
        else:
            classifier = "Blink"

        ##############################################################################################################################################################
        # below is if you want to use the trained idea
        with open('confusionMatrix_all_threshold_0.20.csv', 'a', newline='') as file:
            all_Frames = csv.writer(file)
            all_Frames.writerow([FrameCounter,
                                 "0"])  # open - setting all open rn , we merge - repalce this with the blink to get teh actual data

            with open('confusionMatrix_Blink_threshold_0.20.csv', 'a', newline='') as file:
                blink_Frames = csv.writer(file)
                stringState = " "

                # first test to check if open
                if classifier == "Open":
                    stringState = "Open"

                else:
                    stringState = "Close"

                if testForUserSquinting == False:
                    if classifier == "Open":
                        OpenFrame_COUNTER += 1
                        # print("Open")
                        if hasFrameBlinkHappened:
                            #       print("Blink too short")
                            OpenFrame_COUNTER = 0
                            BlinkFrame_COUNTER = 0
                            potenialBlinkFrame.clear()
                            hasFrameBlinkHappened = False

                    if OpenFrame_COUNTER >= OPEN_EYE_FRAMES_NEEDED:
                        # print("OPEN_EYE_FRAMES_NEEDED passed")
                        if classifier == "Blink":
                            #     print("Blink counter = ", BlinkFrame_COUNTER)
                            hasFrameBlinkHappened = True
                            potenialBlinkFrame.append(FrameCounter)
                            BlinkFrame_COUNTER += 1

                    if BlinkFrame_COUNTER >= BLINK_EYE_FRAMES_NEEDED - 1 and OpenFrame_COUNTER >= OPEN_EYE_FRAMES_NEEDED:
                        testForUserSquinting = True
                else:
                    #    print("Squint test")
                    if classifier == "Open":
                        testForUserSquinting = False
                        #  print("blink detected")

                        tempNumber = len(potenialBlinkFrame)
                        x_range = range(tempNumber)
                        for i in x_range:
                            blink_Frames.writerow([potenialBlinkFrame[i], "1"])  # blink

                        TOTAL += 1
                        totalBlinkFrames = BlinkFrame_COUNTER + totalBlinkFrames
                        OpenFrame_COUNTER = 0
                        BlinkFrame_COUNTER = 0
                        potenialBlinkFrame.clear()

                    else:
                        SquintTestCounter = SquintTestCounter + 1
                        potenialBlinkFrame.append(FrameCounter)
                    #    print("Squint counter = ", SquintTestCounter)
                    if SquintTestCounter < SquintTestCounterThreshold:
                        OpenFrame_COUNTER = 0
                        BlinkFrame_COUNTER = 0
                        potenialBlinkFrame.clear()
                        #     print("Squint test failed")
                        testForUserSquinting = False

            # reset the eye frame counter
        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        tempString = "Current Eye State : " + stringState
        cv2.rectangle(frame, (590, 25), (900, 60), (0, 0, 0), -1)
        cv2.putText(frame, "Total Blinks: {}".format(TOTAL), (600, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ##
        ##
        ##
        cv2.rectangle(frame, (0, 25), (300, 60), (0, 0, 0), -1)
        cv2.putText(frame, "Method : Threshold = 0.2".format(TOTAL), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


    FrameCounter = FrameCounter + 1

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the space bar was pressed, break from the loop
    if key == ord(" "):
        print("Program stopped = ", TOTAL)
        break

# do a bit of cleanup
print("Total BLINK frames: = ", totalBlinkFrames)
print("Total Open frames  = ", FrameCounter - totalBlinkFrames)
print("Total BLINK = ", TOTAL)
cv2.destroyAllWindows()
vs.stop()
