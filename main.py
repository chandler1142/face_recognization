import multiprocessing as mp
import os
from multiprocessing import Pool
from multiprocessing import Process

import cv2
from config.paths import video_path, image_origin, image_ali, csv_dir_indiv, vector_path

from detector import Detector


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


def main():
    print("start face recognization...")
    print("Current host contains cores: ", mp.cpu_count())

    p = Pool(processes=mp.cpu_count())

    detector = Detector()

    video_capture = cv2.VideoCapture(0)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        cv2.putText(frame, "adjust, 'a' to start,'ESC' to quit", (40, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0),
                    0, True)
        cv2.imshow('Capture', frame)
        if cv2.waitKey(50) & 0xFF == ord('a'):
            user_name = input("UserName: ")
            print("Start recoding...")
            record = 0
            user_video_path = video_path + "/" + user_name + ".avi"
            video_writer = cv2.VideoWriter(user_video_path, cv2.VideoWriter_fourcc(*'MJPG'), 30.,
                                           (frame.shape[1], frame.shape[0]))
            cv2.putText(frame, "shake your head, wait 10 seconds to quit", (40, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        (255, 0, 0), 0, True)
            while True:
                record += 1
                ret, frame = video_capture.read()
                video_writer.write(frame)
                cv2.putText(frame, "shake your head, wait 10 seconds to quit", (40, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                            (255, 0, 0), 0, True)
                cv2.imshow('Capture', frame)
                cv2.waitKey(1)
                if record == 180:
                    break
            video_writer.release()
            print("End recording...")
            # 录制结束，就可以通知异步线程去抽取特征样本了
            lock = mp.Lock()
            p = Process(target=detector.extract_user_vectors,
                        args=(user_name, user_video_path, image_origin, image_ali, lock))
            p.start()
        if cv2.waitKey(40) & 0xFF == 27:
            cv2.destroyAllWindows()
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("/------Finished Recording------/")


if __name__ == '__main__':
    init()
    main()
