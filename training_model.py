import os
import pickle
import sys
from cv2 import cv2

import face_recognition


def train_model_by_img(name):
    if not os.path.exists("dataset_from_video"):
        print("[ERROR] there is no directory 'dataset'")
        sys.exit()

    known_encoding = []
    images = os.listdir("dataset_from_video")
    # images = cv2.VideoCapture(0)
    # print(images)

    for (i, image) in enumerate(images):
        print(f"[+]processing img {i + 1}/{len(images)}")
        # print(image)

        face_img = face_recognition.load_image_file(f"dataset_from_video/{image}")
        face_enc = face_recognition.face_encodings(face_img, model = 'cnn')[0]

        # print(face_enc)

        if len(known_encoding) == 0:
            known_encoding.append(face_enc)
        else:
            for item in range(0, len(known_encoding)):
                result = face_recognition.compare_faces([face_enc], known_encoding[item])
                # print(result)

                if result[0]:
                    known_encoding.append(face_enc)
                    # print("Same person")
                    break
                else:
                    # print("Another person")
                    break

    # print(known_encoding)
    # print(f"Length {len(known_encoding)}")
    data = {
        "name": name,
        "encoding": known_encoding
    }

    with open(f"{name}_encoding.pickle", "wb") as file:
        file.write(pickle.dumps(data))

    return f"[INFO] File {name}_encoding.pickle successfully created "


def take_screenshot_from_video():
    cap = cv2.VideoCapture(0)
    count = 0
    if not os.path.exists("dataset_from_video"):
        os.mkdir("dataset_from_video")

    while True:
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        multiplier = fps * 3
        #print(fps)

        if ret:
            frame_id = int(round(cap.get(1)))
            #print(frame_id)
            cv2.imshow("frame", frame)
            cv2.waitKey(28)

            if frame_id % multiplier == 0:
                cv2.imwrite(f"dataset_from_video/{count}screen.jpg", frame)
                print(f"Take a screenshot {count}")
                count += 1

        else:
            print("[Error] Can't get the frame...")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    print(train_model_by_img("Anthony"))
    #print(take_screenshot_from_video())


if __name__ == "__main__":
    main()
