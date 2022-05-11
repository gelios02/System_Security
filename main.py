import face_recognition
from PIL import Image, ImageDraw
import pickle
from cv2 import cv2


def face_rec():
    gal_face_img = face_recognition.load_image_file("img/favotite_actors.jpg")
    gal_face_locations = face_recognition.face_locations(gal_face_img)
    print(f"Found {len(gal_face_locations)} faces")
    print(gal_face_img)

    pil_img1 = Image.fromarray(gal_face_img)
    draw1 = ImageDraw.Draw(pil_img1)

    for (top, right, bottom, left) in gal_face_locations:
        draw1.rectangle(((left, top), (right, bottom)), outline=(255, 255, 0), width=4)

    del draw1
    pil_img1.save("img/AVENGERS_new.jpg")


def extracting_faces(img_path):
    count = 0
    faces = face_recognition.load_image_file(img_path)
    faces_locations = face_recognition.face_locations(faces)

    for faces_locations in faces_locations:
        top, right, bottom, left = faces_locations

        face_img = faces[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f"img/{count}faces.jpg")

        count += 1
    return f"Found {count} faces"


def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encoding = face_recognition.face_encodings(img1)[0]
    # print(img1_encoding)

    img2 = face_recognition.load_image_file(img2_path)
    img2_encoding = face_recognition.face_encodings(img2)[0]

    result = face_recognition.compare_faces([img1_encoding], img2_encoding)
    # print(result)

    if result[0]:
        print("welcome to the club :*")
    else:
        print("Go home")


def detect_person_in_video():
    data = pickle.loads(open("Anthony_encoding.pickle", "rb").read())
    video = cv2.VideoCapture(0)

    while True:
        rer, image = video.read()

        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)

        for face_encoding, face_location in zip(encodings, locations):
            result = face_recognition.compare_faces(data["encoding"], face_encoding)
            match = None

            if True in result:
                match = data["name"]
                print(f"Match found! {match}")
            else:
                print("ALARM!")

            left_top = (face_location[3], face_location[0])
            right_bottom = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, left_top, right_bottom, color, 4)

            left_bottom = (face_location[3], face_location[2])
            right_bottom = (face_location[1], face_location[2] + 20)
            cv2.rectangle(image, left_bottom, right_bottom, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 4)

        cv2.imshow("detect_person_in_video is running", image)
        k = cv2.waitKey(1)
        if k == ord("q"):
            print("Q pressed, closing the app")
            break


def main():
    # face_rec()
    # print(extracting_faces("img/favotite_actors.jpg"))
    # compare_faces("img/img1.jpg", "img/img2.jpg")
    detect_person_in_video()


if __name__ == '__main__':
    main()
