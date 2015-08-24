# -*- coding: utf-8 -*-
__author__ = 'STU_nwad'
import face_process
import face_recognizer
import cv2
import os
import pickle


class FaceInteraction(object):

    Capture_count = 20

    def __init__(self, max_count, name):
        # all data is saved in a folder named name
        self.max_count = max_count  # how many people this instance can recognize
        self.count = 0
        self.person_set = set()
        self.instance_name = name    # used to save the training result
        self.dirname = self.instance_name + os.path.sep
        dirname = self.dirname
        self.train_data = dirname + self.instance_name
        self.dict_data = self.train_data + '_dict'
        self.person_data = dirname + 'persons'
        self.album = dirname + self.instance_name + "_pic"
        self.csv = dirname + self.instance_name + '_csv'
        self.recognizer = cv2.createLBPHFaceRecognizer(1, 8, 8, 8)  # default from c++
        self.dict = None
        if self.exist():
            self.load()
        if self.count > 0:
            print "now I remember ", self.count, "person(s)"
            print 'they are:'
            for people in self.person_set:
                print people,   # no '\n' at the end
            print

    def exist(self):
        if os.path.exists(self.train_data) and os.path.exists(self.dict_data) and os.path.exists(self.person_data):
            return True
        else:
            return False

    def load(self):
        self.recognizer.load(self.train_data)   # training data
        with open(self.dict_data, 'rb') as f:   # dict_data
            self.dict = pickle.load(f)
        with open(self.person_data, 'rb') as f:  # read all persons I have remembered
            self.person_set = pickle.load(f)
        self.count = len(self.person_set)

    def predict(self, img):
        # the img is in grayscale
        label, confidence = self.recognizer.predict(img)
        name = self.dict.get(label, "")
        return name, confidence

    def predict_on_video(self):
        if not self.exist():
            print 'there is no training data, we must train first!'
            return
        videoCap = cv2.VideoCapture(0)
        while True:
            ret, frame = videoCap.read()
            if ret:
                frame = cv2.flip(frame, 1)  # horizontal
                face_imgs, face_coords = face_process.cropFaces(frame, grayscale=True)
                if len(face_imgs) != 0:
                    # outline each face detected
                    for face_area in face_coords:
                        face_process.outline(frame, face_area, (255, 0, 0))

                    face = face_imgs[0]     # now just consider one person
                    face_coord = face_coords[0]
                    name, confidence = self.predict(face)
                    x, y = face_coord[:2]
                    if confidence <= 65:
                        print "I guess you are ", name, "^_^", confidence
                        face_process.drawText(frame, name, (x, y - 30))
                    else:
                        face_process.drawText(frame, "Nobody", (x, y - 30))
                        print "I'm sorry! I guess I don't know you...But we can be friends~"
            cv2.imshow(self.instance_name, frame)
            key = cv2.waitKey(50)
            if key == ord('q'):
                break

    def get_faces(self, name):
        face_process.process_dir(self.dirname + name,
                                 folder_for_each_picture=False, delete_src=True)    # delete the original picture

    def train(self):
        # first record the photos
        name = ""
        while name == "":
            name = raw_input('input your name(^_^):\n')
        self.person_set.add(name)
        # update the persons
        with open(self.person_data, 'wb') as f:
            pickle.dump(self.person_set, f)
        print "start recording your faces~~~~"
        count = FaceInteraction.Capture_count
        face_process.record_face(count, self.dirname + name)
        print 'recording ends.{} photos of you are token~'.format(count)
        print 'get all the faces from the pictures~~~'
        self.get_faces(name)
        print 'start creating csv file for you~'
        csv = face_recognizer.creat_csv(self.dirname, commentNamedByFolderName=True)
        # print csv
        with open(self.csv, 'w') as f:
            f.writelines(csv)
        print 'csv file created!'
        print 'start training'
        face_recognizer.train(self.csv, self.train_data)


if __name__ == "__main__":
    demo = FaceInteraction(10, 'nwad')  # 10 people at most, 'nwad' is the name of the folder the program uses
    demo.predict_on_video()
    # demo.predict_on_video()
