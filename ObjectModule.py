import cv2
import numpy as np
import time
import datetime


class ObjectDetector:

    def __init__(self, outputs, img, objectcount, confThreshold, nsmThreshold , classNames):
        self.outputs = outputs
        self.img = img
        self.objectcount = objectcount
        self.confThreshold = confThreshold
        self.nsmThreshold = nsmThreshold
        self.classNames = classNames

    def findobjects(self):
            """ height , widht and channel of image"""
            hT, wT, cT = self.img.shape
            bbox = []
            classIds = []
            confs = []

            for output in self.outputs:
                for det in output:
                    scores = det[5:]
                    # print(scores)
                    classId = np.argmax(scores)
                    #print(classId)
                    confidence = scores[classId]
                    if confidence > self.confThreshold:
                        w, h = int(det[2] * wT), int(det[3] * hT)
                        x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                        bbox.append([x, y, w, h])
                        self.objectcount += 1
                        classIds.append(classId)
                        confs.append(float(confidence))

            # print(len(bbox))
            """ suppress the non maximum confidence boxes"""
            indices = cv2.dnn.NMSBoxes(bbox, confs, self.confThreshold, self.nsmThreshold)
            # print(type(indices))
            for i in indices:
                # i = i[0]
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(self.img, f'{self.classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                            (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 255), 2)

            return self.objectcount


