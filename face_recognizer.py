# -*- coding: utf-8 -*-
__author__ = 'STU_nwad'
# this file is translated from c++ to python
# /*
#  * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
#  * Released to public domain under terms of the BSD Simplified license.
#  *
#  * Redistribution and use in source and binary forms, with or without
#  * modification, are permitted provided that the following conditions are met:
#  *   * Redistributions of source code must retain the above copyright
#  *     notice, this list of conditions and the following disclaimer.
#  *   * Redistributions in binary form must reproduce the above copyright
#  *     notice, this list of conditions and the following disclaimer in the
#  *     documentation and/or other materials provided with the distribution.
#  *   * Neither the name of the organization nor the names of its contributors
#  *     may be used to endorse or promote products derived from this software
#  *     without specific prior written permission.
#  *
#  *   See <http://www.opensource.org/licenses/bsd-license>
#  */
import glob
import cv2
import numpy
import sys
import os
import pickle
# import argparse

TRAIN_RESUT = "train_result"
model = cv2.createLBPHFaceRecognizer(radius=1, neighbors=8, grid_x=8, grid_y=8)  # the default value from c++


def read_csv(filename, images, labels, labelsInfo, sep=';'):
    """
    :param filename: the csv file format <path>;<label>[;<comment>]\n
    :param images: where to store cv2 Image
    :param labels: the labels associated with the Img
    :param labelsInfo: like name or some other information about the Image
    :param sep: how to split the line in csv file
    :return: None
    """

    with open(filename, 'r') as csv_file:
        all_lines = csv_file.readlines()
        for line in all_lines:
            # <path>;<label>[;<comment>]\n
            path = label = comment = ""
            if line.count(sep) == 2:
                path, label, comment = line.split(sep)
            elif line.count(sep) == 1:
                path, label = line.split(sep)
            comment = comment.strip()
            assert isinstance(path, str)
            assert isinstance(label, str)
            assert isinstance(comment, str)
            if len(path) != 0 and len(label) != 0:
                print "Processing", path
                label_num = int(label)
                labelsInfo[label_num] = comment
            # 'path' can be file, dir or wildcard path
            files = glob.glob(path)
            for each_file in files:
                img = cv2.imread(each_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)    # grayscale pictures
                assert isinstance(img, numpy.ndarray)
                height, width = img.shape[:2]
                if width < 50 or height < 50:
                    print 'Warning: * Warning: for better results images should be not smaller than 50x50!'
                images.append(img)
                labels.append(label_num)


def train(csv_file, filenameTosave=TRAIN_RESUT):
    # csv_file = sys.argv[1]
    images = list()
    labels = list()
    labelsInfo = dict()
    try:
        read_csv(csv_file, images, labels, labelsInfo)
    except Exception as err:
        print "Failed to open file", csv_file, '\n', str(err)
        sys.exit(1)

    if len(images) <= 1:
        print "need at least 2 images"
        sys.exit(2)

    # The following lines simply get the last images from
    #  your dataset and remove it from the vector. This is
    #  done, so that the training data (which we learn the
    #  cv::FaceRecognizer on) and the test data we test
    #  the model with, do not overlap.
    print labelsInfo
    testImg = images[len(images) - 1]
    testLabel = labels[len(labels) - 1]
    images.pop(len(images) - 1)
    labels.pop(len(labels) - 1)

    global model

    # for key in labelsInfo:
    #     assert isinstance(model, cv2.FaceRecognizer)
    #     model.setString(str(key), str(labelsInfo[key]))
    print 'start training'
    model.train(images, numpy.array(labels))
    model.save(filenameTosave)
    # save the dict of name to label
    with open(filenameTosave + "_dict", 'wb') as f:
        pickle.dump(labelsInfo, f)
    print 'the training result is saved as', \
        filenameTosave
    # The smaller the precise of confidence is
    label, confidence = model.predict(testImg)
    print "prediction label is {} and the info is {}".format(label, labelsInfo[label])
    print "actual labe is     ", testLabel
    print 'the confidence is  ', confidence


def predict(filenames, train_result, print_out=False):
    """
    :param filenames: the image reading from cv2
    :param train_result: the training_result
    :return: filename, label, confidence
    """
    # model = cv2.createLBPHFaceRecognizer(1, 8, 8, 8)  # default from c++
    global model
    model.load(train_result)
    # labelsInfo = dict()
    try:
        with open(train_result + "_dict", 'rb') as f:
            labelsInfo = pickle.load(f)
    except IOError as err:
        labelsInfo = dict()
    result = []
    if print_out:
        print "file\tlabel[name]\tconfidence"
    for each_file in filenames:
        # print "loading", each_file
        img = cv2.imread(each_file, cv2.CV_LOAD_IMAGE_GRAYSCALE)  # must be grayscale and the same size
        label, confidence = model.predict(img)
        if print_out:
            print '\t' + os.path.basename(each_file), "{}[{}]".format(label, labelsInfo.get(label, "")), confidence
        result.append((each_file, label, confidence))
    return result


def usage():
    print "Usage:\ntrain: " + sys.argv[0] + " -t <csv>"
    print "predict: " + sys.argv[0] + " -p <train_result> [imgfile]"
    print "predict: " + sys.argv[0] + " -pd <train_result> [imgfolder <pattern>]"
    print "creatCSV file: " + sys.argv[0] + " -c <folder>"
    print "Notes: " \
          'The CSV config file consists of the following lines:\n' \
          '\t<path>;<label>[;<comment>]\n' \
          '\t<path> - file, dir or wildcard path\n' \
          '\t<label> - non-negative integer person label\n' \
          '\t<comment> - optional comment string (e.g. person name)'
    sys.exit(2)


def predict_dir(root='.', pattern="*.jpg", train_result=TRAIN_RESUT):
    """
    predict each pattern file and return the result (filename, label, confidence)
    :param root: the root dir
    :param pattern: can be wildcard
    :return: list(filename, label, confidence)
    """
    root = os.path.abspath(root)
    path = os.path.join(root, pattern)
    filenames = glob.glob(path)
    predict(filenames, train_result, True)


def argparser():
    if len(sys.argv) == 1:
        usage()
    valid = ('-t', '-p', '-c')
    if sys.argv[1] in valid and len(sys.argv) < 3:
        usage()
    if sys.argv[1] == '-t':
        csv_file = sys.argv[2]
        train(csv_file)
    elif sys.argv[1] == '-p':
        train_result = sys.argv[2]
        if len(sys.argv) >= 4:
            imgfile = sys.argv[3]
            predict([imgfile], train_result, True)
            return
        else:
            # ask user to input
            pass
    elif sys.argv[1] == '-pd':
        train_result = sys.argv[2]
        if len(sys.argv) == 4:
            folder = sys.argv[3]
            predict_dir(folder, train_result=train_result)
        else:
            predict_dir(sys.argv[3], sys.argv[4], train_result=train_result)
    elif sys.argv[1] == '-c':
        creat_csv(sys.argv[2], commentNamedByFolderName=True)

    # parser = argparse.ArgumentParser(description="create csv file;train the recognizer;predict with the training result")
    # group = parser.add_mutually_exclusive_group()
    # group.add_argument('-t', '--train', help="specify a csv file to train your recognizer"
    #                                          "\ncsv file format:\n<path>;<label>[;<comment>]\\n")
    # group.add_argument('-p', '--predict', help="")
    # parser.add_argument('training_result', help="the file contains the training result")
    # args = parser.parse_args()


def creat_csv(path=".", comment='', sep=';', pattern=(".jpg", ".png"), commentNamedByFolderName=False):
    if not os.path.isdir(path):
        print "the argument must be a dir name"
        sys.exit(3)
    count = 0
    lines = []
    for cwd, dirs, files in os.walk(os.path.abspath(path)):
        # print files
        for each_file in files:  # picture in the same folder has the same label
            if os.path.splitext(each_file)[1] in pattern:   # filter
                line = ""
                if not commentNamedByFolderName:
                    line = os.path.join(cwd, each_file) + sep + str(count) + sep + comment
                elif commentNamedByFolderName and comment == "":
                    line = os.path.join(cwd, each_file) + sep + str(count) + sep + os.path.basename(cwd)
                # print line
                lines.append(line + '\n')
        count += 1
    return lines

if __name__ == "__main__":
    argparser()
