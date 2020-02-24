import csv

import cv2
import dlib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from config.paths import csv_dir_indiv
from imutils import face_utils

from openface_pytorch import netOpenFace

# 人脸image提取
model = './models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=250)

transform = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


class Detector(object):

    def __init__(self, useCuda=False, gpuDevice=0, useMultiGPU=True):
        super(Detector, self).__init__()
        # self.facenet = netOpenFace(useCuda, gpuDevice)
        # # model.load_state_dict(torch.load('/home/zouzijie/Desktop/Particial_Facenet/FaceRec_PyTorch_V1.3/models/openface_nn4_small2_v1.pth'))
        # self.facenet.load_state_dict(
        #     torch.load('./models/openface_20180119.pth', map_location=lambda storage, loc: storage))
        # if useMultiGPU:
        #     self.facenet = nn.DataParallel(self.facenet)
        # if torch.cuda.is_available():
        #     self.facenet = self.facenet.cuda()

    def __write_basic_csv(self, counter, user_name, Ori_path, Ali_path):
        dir = csv_dir_indiv + '/' + str(user_name) + '.csv'
        with open(dir, 'a', newline='') as wf1:
            writer1 = csv.writer(wf1)
            header1 = [counter, Ori_path, Ali_path, 'Null']
            writer1.writerow(header1)

    def __prepare_images(self, user_name, user_video_path, image_origin, image_ali):
        capture = cv2.VideoCapture(user_video_path)
        counter = 0
        while counter < 150:
            ret, frame = capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rects = detector(gray, 1)

            for (i, rect) in enumerate(rects):
                faceAligned = fa.align(frame, gray, rect)
                ali_path, ori_path = "", ""
                if counter < 10:
                    ori_path = image_origin + '/' + user_name + '_000' + str(counter) + '.jpg'
                    ali_path = image_ali + '/' + user_name + '_000' + str(counter) + '.jpg'
                elif 10 <= counter < 100:
                    ori_path = image_origin + '/' + user_name + '_00' + str(counter) + '.jpg'
                    ali_path = image_ali + '/' + user_name + '_00' + str(counter) + '.jpg'
                elif 100 <= counter < 1000:
                    ori_path = image_origin + '/' + user_name + '_0' + str(counter) + '.jpg'
                    ali_path = image_ali + '/' + user_name + '_0' + str(counter) + '.jpg'
                cv2.imwrite(ali_path, faceAligned)
                self.__write_basic_csv(counter, user_name, ori_path, ali_path)
                counter += 1
            print("Captured image : " + user_name + str(counter))

    def __extract_from_images(self):
        pass

    def extract_user_vectors(self, user_name, user_video_path, image_origin, image_ali):
        print("start to extract_user_vectors...")
        self.__prepare_images(user_name, user_video_path, image_origin, image_ali)
        self.__extract_from_images()
        # train()
