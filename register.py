import csv
import os

import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from config.paths import csv_dir_indiv, vector_path
from imutils import face_utils
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from torch.autograd import Variable

from openface_pytorch import netOpenFace

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


class Register(object):

    def __init__(self):
        super(Register, self).__init__()

    def __write_basic_csv_data(self, user_name, basic_csv_data):
        dir = csv_dir_indiv + '/' + str(user_name) + '.csv'
        if os.path.exists(dir):
            os.remove(dir)
        with open(dir, 'a', newline='') as wf1:
            writer1 = csv.writer(wf1)
            writer1.writerows(basic_csv_data)

    def __write_basic_csv(self, counter, user_name, Ori_path, Ali_path):
        dir = csv_dir_indiv + '/' + str(user_name) + '.csv'
        if os.path.exists(dir):
            os.remove(dir)
        with open(dir, 'a', newline='') as wf1:
            writer1 = csv.writer(wf1)
            header1 = [counter, Ori_path, Ali_path, 'Null']
            writer1.writerow(header1)

    def __prepare_openface(useCuda=False, gpuDevice=0, useMultiGPU=False):
        model = netOpenFace(useCuda, gpuDevice)
        model.load_state_dict(torch.load('./models/openface_20180119.pth', map_location=lambda storage, loc: storage))
        if useMultiGPU:
            model = nn.DataParallel(model)
        if torch.cuda.is_available():
            return model.cuda()
        return model

    def __prepare_images(self, user_name, user_video_path, image_origin, image_ali):
        capture = cv2.VideoCapture(user_video_path)
        counter = 0
        basic_csv_data = []
        while counter < 10:
            ret, frame = capture.read()

            if frame is None or len(frame) <= 0:
                break

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
                counter += 1
                print("Captured image : " + user_name + str(counter))
                basic_csv_data.append((counter, ori_path, ali_path, 'Null'))

        self.__write_basic_csv_data(user_name, basic_csv_data)
        print("prepare images finished...")

    def __extract_from_images(self, user_name):
        dir = csv_dir_indiv + '/' + str(user_name) + '.csv'
        facenet = self.__prepare_openface()
        with open(dir, 'r', ) as rf:
            reader = list(csv.reader(rf))
            num = len(reader)

        index = 0

        for row in reader:
            # row ['9', './data/images_ori//ZhaoWei_0009.jpg', './data/images_ali//ZhaoWei_0009.jpg', 'Null']
            image_ali_path = row[2]

            img_pil = Image.open(image_ali_path)
            img_tensor = transform(img_pil)
            img_tensor = self.__to_var(img_tensor)
            outputs = facenet(img_tensor.unsqueeze(0))

            if index < 10:
                vector_temp = vector_path + '/' + user_name + '_000' + str(index) + '.npy'
                np.save(vector_temp, self.__to_np(outputs[0]))

            elif 10 <= index < 100:
                vector_temp = vector_path + '/' + user_name + '_00' + str(index) + '.npy'
                np.save(vector_temp, self.__to_n(outputs[0]))

            elif 100 <= index < 1000:
                vector_temp = vector_path + '/' + user_name + '_0' + str(index) + '.npy'
                np.save(vector_temp, self.__to_n(outputs[0]))

            row[3] = vector_temp
            with open(dir, 'w', newline='') as wf:
                writer1 = csv.writer(wf)
                writer1.writerows(reader)

            index += 1

        print("extract images finished...")
        return dir

    def __train(self, user_name, csv_path):
        X_train, Y_train = self.__generate_dataset(user_name, csv_path)
        knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
        knn.fit(X_train, Y_train)
        joblib.dump(knn, './models/knn.model')
        print("train finished...")

    def __generate_dataset(self, user_name, csv_path):
        x_train = []
        y_train = []

        rf = open(csv_path, 'r')
        reader = list(csv.reader(rf))
        counter = 0
        for k in reader:
            if counter > 0:
                read_path = k[3]
                data = self.__falttern(read_path)
                x_train.append(data)
                y_train.append(user_name)
            else:
                pass
            counter += 1
        return x_train, y_train

    def __falttern(self, path):
        vector = np.load(path)
        result = vector.flatten()
        return result

    def __to_np(self, x):
        return x.data.cpu().numpy()

    def __to_var(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def extract_user_vectors(self, user_name, user_video_path, image_origin, image_ali, lock):
        print("start to extract_user_vectors...")
        lock.acquire()
        self.__prepare_images(user_name, user_video_path, image_origin, image_ali)
        csv_path = self.__extract_from_images(user_name)
        self.__train(user_name, csv_path)
        lock.release()
