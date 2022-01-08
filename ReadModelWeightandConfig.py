import cv2
import numpy as np
import time
import datetime


class yolov3network:
    def __init__(self, modelConfiguration, modelWeights):
        self.modelConfiguration = modelConfiguration
        self.modelWeights = modelWeights

    def readnetwork(self):
        net = cv2.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        return net
