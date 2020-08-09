'''
Uses weights and models implementation' from
https://github.com/deepinsight/insightface
'''

import numpy as np
import argparse
from os import path, listdir, makedirs
import cv2
import sys
sys.path.insert(0, '../insightface/deploy/')
import face_model
from sklearn.metrics.pairwise import cosine_similarity
from os import path, makedirs
from multiprocessing import Pool
from scipy.spatial import distance
import glob

def chisquare(p, q):
    p = np.asarray(p).flatten()
    q = np.asarray(q).flatten()
    bin_dists = (p - q)**2 / (p + q + np.finfo('float').eps)
    return np.sum(bin_dists)

def match(features_a,features_b,METRIC):
    if METRIC == 1:
        score = np.mean(cosine_similarity(features_a, features_b))
    elif METRIC == 2:
        score = distance.euclidean(features_a, features_b)
    else:
        score = chisquare(features_a, features_b)
    return score


def extract_features(model,img):
    img, cropped = model.get_input(img)
    cv2.imshow("ok", cropped)
    cv2.waitKey(0)
    features = model.get_feature(img)
    return features

def read_image_from_folder(folder_name):
    files = glob.glob(folder_name + "/*")
    img1 = cv2.imread(files[0])
    img2 = cv2.imread(files[1])
    return img1,img2

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract features with CNN')
    parser.add_argument('--folder', '-f', help='image folder')
    parser.add_argument('--thresh', '-th', default=1 , help='threshold')
    # parser.add_argument('--img1', '-i1', help='id document')
    # parser.add_argument('--img2', '-i2', help='selfie image')
    # ArcFace params
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', help='path to model.', default='../../insightface/models/model-r100-ii/model,0')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument('--gender_model', default='', help='path to load model.')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=1, type=int, help='mtcnn: 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')

    args = parser.parse_args()
    model = face_model.FaceModel(args)
    while True:
        img1, img2 = read_image_from_folder(args.folder)
        # img1 = cv2.imread(args.img1)
        # img2 = cv2.imread(args.img2)
        print("Thresh:", float(args.thresh))
        start = time.time()
        feature1 = extract_features(model, img1)
        feature2 = extract_features(model, img2)
        end = time.time()
        score = match(feature1, feature2, 2)
        print("Distance:", score)
        if score < float(args.thresh):
            print("The same person")
        else:
            print("Different person")
        print("Total time:", end - start)
        print("_____________________________________")
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        else:
            continue
