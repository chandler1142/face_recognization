import multiprocessing as mp
import os
from multiprocessing import Pool
from multiprocessing import Process

import cv2
from config.paths import video_path, image_origin, image_ali, csv_dir_indiv, vector_path

from knn_model import knn_classifier
from register import Register, transform
import dlib
from imutils import face_utils
from PIL import Image

import time

from utils import to_var, to_np


def init():
    print("Prepare running env...")

    if not os.path.exists(os.path.join(video_path)):
        os.mkdir(os.path.join(video_path))
    if not os.path.exists(os.path.join(image_origin)):
        os.mkdir(os.path.join(image_origin))
    if not os.path.exists(os.path.join(image_ali)):
        os.mkdir(os.path.join(image_ali))
    if not os.path.exists(os.path.join(csv_dir_indiv)):
        os.mkdir(os.path.join(csv_dir_indiv))
    if not os.path.exists(os.path.join(vector_path)):
        os.mkdir(os.path.join(vector_path))


detector = dlib.get_frontal_face_detector()

model = './models/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)
fa = face_utils.FaceAligner(predictor, desiredFaceWidth=250)


def main():
    print("start face recognization...")
    print("Current host contains cores: ", mp.cpu_count())
    register = Register()

    video_capture = cv2.VideoCapture(0)
    while video_capture.isOpened():
        ret, frame = video_capture.read()

        # 识别
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b_box = []
        rects = detector(gray, 1)
        frame_index = 0
        total_s = time.time()

        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)

            faceAligned = fa.align(frame, gray, rect)
            image = cv2.cvtColor(faceAligned, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(image)

            # /---compute vectors and do KNN---/
            img_tensor = transform(pil_im)
            img_tensor = to_var(img_tensor)
            outputs_128, outputs_726 = register.facenet(img_tensor.unsqueeze(0))
            outputs = to_np(outputs_128)

            outputs = outputs.flatten().reshape(1, -1)

            pred, prob = knn_classifier(outputs, './models/knn.model')

            name = pred[0]
            b_box.append(((x, y), (x + w, y + h), name))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), 1, True)

            print("Frame=" + str(frame_index) + "---" + name)

        Intro = "'a' to start,'ESC' to quit"
        cv2.putText(frame, Intro, (40, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 0, True)
        cv2.imshow("Rec Recoginition", frame)

        total_e = time.time()
        print("total=", total_e - total_s)

        frame_index += 1

        if cv2.waitKey(50) & 0xFF == ord('a'):
            user_name = input("UserName: ")
            print("Start recoding...")
            record = 0
            user_video_path = video_path + "/" + user_name + ".avi"
            video_writer = cv2.VideoWriter(user_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 30.,
                                           (frame.shape[1], frame.shape[0]))

            while True:
                ret, frame = video_capture.read()

                Intro = "Shake your head"
                cv2.putText(frame, Intro, (40, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 0, True)
                cv2.imshow("Rec Recoginition", frame)

                record += 1
                video_writer.write(frame)
                cv2.waitKey(1)
                if record == 180:
                    break
            video_writer.release()
            print("End recording...")
            # 录制结束，就可以通知异步线程去抽取特征样本了

            import threading
            # register.extract_user_vectors(user_name, user_video_path, image_origin, image_ali)

            t = threading.Thread(target=register.extract_user_vectors, args=(user_name, user_video_path, image_origin, image_ali))
            t.start()
        if cv2.waitKey(40) & 0xFF == 27:
            cv2.destroyAllWindows()
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("/------Finished Recording------/")


if __name__ == '__main__':
    init()
    main()
