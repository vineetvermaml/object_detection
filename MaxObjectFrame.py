import cv2
import numpy as np
import time
import datetime
from ObjectModule import *
from ReadClassFile import *
from ReadModelWeightandConfig import *

"""width and height of image"""
whT = 320
""" minimum confidence threshold"""
confThreshold = 0.5
nsmThreshold = 0.2
""" no of objects per frame"""
objectcount = 0
pTime = 0

CLASSESFILE = 'ClassName/objectsname.txt'
MODELCONFIGURATION = 'WeightConfigNetworkFiles/yolov3-tiny.cfg'
MODELWEIGHTS = 'WeightConfigNetworkFiles/yolov3-tiny.weights'

framedetails = {}
classNames = []
cap = cv2.VideoCapture("videos/video02.mp4")


""" Name of the classes model will extract or identify """
readclassfile = ReadClass(CLASSESFILE)
classNames = readclassfile.readFileasList()

""" Reading the network weights and Configuration"""
yolov3network = yolov3network(MODELCONFIGURATION, MODELWEIGHTS)
net = yolov3network.readnetwork()
""" Using local machine hence setting the preference as CPU"""
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while True:
    """ Read each frame from Video"""
    success, img = cap.read()
    if not success:
        print("Can't receive frame (stream end?). Exiting ...")
        """ reading of all frames are completed, find the frame with maximum objects detected
        and display the frame name and frame number"""
        MaxKey = max(framedetails, key=framedetails.get)
        print(f"Details of all frames{framedetails}")
        print(f'This Frame {MaxKey} detected maximum object {framedetails[MaxKey]}')
        break
    cTime = time.time()  # returns epoch time
    cTime = datetime.datetime.fromtimestamp(cTime)  # change to yyyy mm dd hh mm ss format
    """ Displaying datetime  """
    cv2.putText(img, f'Frame:{cTime}', (10, 40), cv2.FONT_HERSHEY_PLAIN,
                1, (0, 255, 0))
    """ Converting the frame into a blob for further processing"""
    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    """ passing the blob in yolov3 network"""
    net.setInput(blob)
    layerNames = net.getLayerNames()
    """ Extract only the output layers """
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()]
    """ Send these layers as forward pass to network"""
    outputs = net.forward(outputNames)
    detector = ObjectDetector(outputs, img, objectcount, confThreshold, nsmThreshold , classNames)
    """ saving the number of object detected per frame"""
    objectcount_new = detector.findobjects()
    """ Using Dictionary , mapping frame name with object detected in frame """
    framedetails[f'{cTime}'] = objectcount_new
    cv2.imshow("image", img)
    cv2.waitKey(1)
