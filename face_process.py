# -*- coding: UTF-8 -*-
__author__ = 'OpenCv tutorial'
import cv2
import os
import glob
import time
Face_Cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml")
Eye_Cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
Eye_With_Glass_Cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye_tree_eyeglasses.xml")


def widthHeightDivideBy(image, divisor):
    """ Return an image's dimensions, divided by a value. """
    h, w = image.shape[:2]
    return w / divisor, h / divisor


def detect_obj(image, classfier, searchArea=None, factor=1.1, minValue=3, flag=cv2.cv.CV_HAAR_SCALE_IMAGE,
               imageSizeToMinSizeRatio=8):
    """
    detect obj use classfier and the specified arguments
    :param image: read from cv2
    :param classfier:
    :param searchArea: (x, y, w, h)
    :param factor:
    :param minValue:
    :param flag:
    :return:
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
    image = cv2.equalizeHist(image)
    minSize = widthHeightDivideBy(image, imageSizeToMinSizeRatio)
    if searchArea is not None:
        # print searchArea
        x, y, w, h = searchArea
        sub_img = image[y: y + h, x: x + w]

    else:
        sub_img = image
    # showDebugImg(sub_img)
    result = classfier.detectMultiScale(sub_img, factor, minValue, flag, minSize)
    if searchArea is not None:
        for i in range(len(result)):
            tx, ty, tw, th = result[i]
            tx += x  # to original coordinate
            ty += y
            result[i] = tx, ty, tw, th
    return result


def showDebugImg(img, area=None):
    if area is not None:
        x, y, w, h = area
        cv2.imshow("debug", img[y: y + h, x: x + w])
        cv2.waitKey(0)
    else:
        cv2.imshow("debug", img)
        cv2.waitKey(0)


def any_faces(image):
    faces = detect_obj(image, Face_Cascade, imageSizeToMinSizeRatio=8)
    result = []
    for face in faces:
        # showDebugImg(image, face)
        x, y, w, h = face
        upper_face = x, y, w, h / 2
        eyes = detect_obj(image, Eye_Cascade, upper_face, imageSizeToMinSizeRatio=64)

        if len(eyes) == 0:
            eyes = detect_obj(image, Eye_With_Glass_Cascade, upper_face, imageSizeToMinSizeRatio=64)
        if len(eyes) != 0:  # can't find eyes
            result.append(face)
    # for face in result:
    #     showDebugImg(image, face)
    return result


def getRect(image, area):
    x, y, w, h = area
    return image[y: y + h, x: x + w]


def cropFaces(image, size=(100, 100), grayscale=False):
    faces = any_faces(image)
    face_rois = []
    for area in faces:
        face = getRect(image, area)
        face = cv2.resize(face, size, interpolation=cv2.INTER_CUBIC)
        if grayscale:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_rois.append(face)
        # showDebugImg(face)
    return face_rois, faces


def process_dir(dirpath=".", pattern="*.jpg", folder_for_each_picture=True, delete_src=False):
    count = 0
    # print dirpath
    # print dirpath + os.path.sep + pattern
    # raw_input('enter something')
    for filename in glob.glob(dirpath + os.path.sep + pattern):
        if folder_for_each_picture:
            dirpath = os.path.splitext(filename)[0]
        elif dirpath == ".":
            dirpath = os.path.basename(os.path.abspath(dirpath))

        img = cv2.imread(filename)
        faces = cropFaces(img)[0]
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        if folder_for_each_picture:
            count = 0   # the count starts over
        for face in faces:
            cv2.imwrite(dirpath + '/' + str(count) + '.jpg', face)
            count += 1
        if delete_src:  # avoid the original picture being used to train the model
            os.remove(filename)


def outline(img, area, color=(255, 0, 0)):
    x, y, w, h = area
    p1 = (x, y)
    p2 = (x + w, y + h)
    cv2.rectangle(img, p1, p2, color)


def drawText(img, text, bottom_left_point, fontface=cv2.FONT_HERSHEY_COMPLEX, fontscale=1,
             color=(255, 255, 255), thinkness=1, line_type = cv2.CV_AA):
    cv2.putText(img, text, bottom_left_point, fontface,
                fontscale, color, thinkness, line_type)


def record_face(amount=-1, dirname=None):
    """
    when two eyes are detected, record automatically
    :return:
    """
    count = 0
    interval = 0
    capture = cv2.VideoCapture(0)
    if dirname is None:
        dirname = str(time.time())
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    while True:
        interval += 1
        ret_val, frame = capture.read()
        if ret_val:
            frame = cv2.flip(frame, 1)
            cv2.imshow('', frame)
            if interval % 10 == 0:
                faces = any_faces(frame)
                if len(faces) != 0:
                    cv2.imwrite(dirname + os.path.sep + 'record_' + str(count) + '.jpg', frame)
                    outline(frame, faces[0])
                    cv2.imshow('', frame)
                    print("capturing {}".format(count))
                    if amount != -1:
                        print "{} left".format(amount - count - 1)
                    count += 1
        key = cv2.waitKey(50)
        if key == 113 or count == amount:   # q
            break


def draw_face_and_put_text(img):
    faces = any_faces(img)
    for face in faces:
        outline(img, face)
        x, y = face[:2]
        drawText(img, 'people', (x, y - 20), fontface=cv2.FONT_HERSHEY_DUPLEX, thinkness=1, color=(14, 24, 34))
        showDebugImg(img)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "-r":
            print('starting recording faces')
            record_face()
    else:
        process_dir()
        sys.exit(0)
    # img = cv2.imread("What Lucy Sunny Loy.jpg")
    # cropFaces(img)
    # any_faces(img)
    # record_face()
    # process_dir(folder_for_each_picture=False)
    # img = cv2.imread("What Lucy Sunny Loy.jpg")
    # draw_face_and_put_text(img)
