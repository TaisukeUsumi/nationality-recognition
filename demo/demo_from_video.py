import argparse
import os

import cv2
import numpy as np
from keras import backend as K
from keras.models import load_model


def draw_label(image,
               point,
               label,
               font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1,
               thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0),
                  cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255),
                thickness)


def draw_results(detected, input_img, faces, add_ratio, img_size, img_w, img_h,
                 model):
    for i, (x, y, w, h) in enumerate(detected):
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h

        xw1 = max(int(x1 - add_ratio * w), 0)
        yw1 = max(int(y1 - add_ratio * h), 0)
        xw2 = min(int(x2 + add_ratio * w), img_w - 1)
        yw2 = min(int(y2 + add_ratio * h), img_h - 1)

        faces[i, :, :, :] = cv2.resize(input_img[yw1:yw2 + 1, xw1:xw2 + 1, :],
                                       (img_size, img_size))

        faces[i, :, :, :] = cv2.normalize(faces[i, :, :, :],
                                          None,
                                          alpha=0,
                                          beta=255,
                                          norm_type=cv2.NORM_MINMAX)

        cv2.rectangle(input_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(input_img, (xw1, yw1), (xw2, yw2), (0, 0, 255), 2)

    if len(detected) > 0:
        result = model.predict(faces)

        for i, (x, y, w, h) in enumerate(detected):
            x1 = x
            y1 = y

            nationality_result = np.argmax(result)

            if nationality_result == 0:
                nationality = 'Chinese'
            elif nationality_result == 1:
                nationality = 'Japanese'
            else:
                nationality = 'Korean'

            label = '{}'.format(nationality)

            draw_label(input_img, (x1, y1), label)

        cv2.imshow('result', input_img)


def demo_from_video(arguments):
    K.set_learning_phase(0)

    face_cascade = cv2.CascadeClassifier(arguments.face_detection_model)
    model = load_model(arguments.nationality_recognition_model)

    if arguments.use_camera:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(arguments.input_video)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024 * 1)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768 * 1)

    img_idx = 0
    skip_frame = 5
    add_ratio = 0.15

    while True:
        ret, input_img = cap.read()
        img_idx += 1

        if len(np.shape(input_img)) == 3:
            img_h, img_w, _ = np.shape(input_img)
        else:
            continue

        if img_idx == 1 or img_idx % skip_frame == 0:
            gray_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray_img, 1.1)
            faces = np.empty(
                (len(detected), arguments.image_size, arguments.image_size, 3))
            draw_results(detected, input_img, faces, add_ratio,
                         arguments.image_size, img_w, img_h, model)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--input_video',
                        type=str,
                        default='',
                        help='Input image data')
    parser.add_argument('--use_camera',
                        action='store_true',
                        help='Predict from camera')
    parser.add_argument('--image_size',
                        type=int,
                        default=128,
                        help='Model input image size')
    parser.add_argument('--nationality_recognition_model',
                        type=str,
                        default='./model/nationality_recognition_model.h5',
                        help='Model file for nationality recognition')
    parser.add_argument('--face_detection_model',
                        type=str,
                        default='./model/lbpcascade_frontalface_model.xml',
                        help='Model file for face detection')
    arguments = parser.parse_args()

    assert os.path.isfile(arguments.input_video)\
           or arguments.use_camera, "Input video data or use --use_camera"

    demo_from_video(arguments)
