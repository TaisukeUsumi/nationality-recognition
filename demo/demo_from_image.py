import argparse
import os

import cv2
import numpy as np
from keras.models import load_model


def draw_label(image,
               point,
               label,
               font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1,
               thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y, w, h = point
    cv2.rectangle(image, (x, y - size[1] - 10), (x + size[0], y), (255, 0, 0),
                  cv2.FILLED)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), thickness)
    cv2.putText(image, label, (x, y - 10), font, font_scale, (255, 255, 255),
                thickness)


def demo_from_image(arguments):
    image = cv2.imread(arguments.input_image)
    original_image = image.copy()

    face_cascade = cv2.CascadeClassifier(arguments.face_detection_model)
    face = face_cascade.detectMultiScale(image)
    for x, y, w, h in face:
        image = image[y:y + h, x:x + w]
    facial_image = cv2.resize(image,
                              (arguments.image_size, arguments.image_size))
    facial_image = np.reshape(
        facial_image, (1, arguments.image_size, arguments.image_size, 3))

    model = load_model(arguments.nationality_recognition_model)
    result = model.predict(facial_image)

    nationality_result = np.argmax(result)

    if nationality_result == 0:
        nationality = 'Chinese'
    elif nationality_result == 1:
        nationality = 'Japanese'
    else:
        nationality = 'Korean'

    label = '{}'.format(nationality)

    for x, y, w, h in face:
        draw_label(original_image, (x, y, w, h), label)
    cv2.imwrite(arguments.output, original_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--input_image',
                        type=str,
                        default='',
                        help='Input image data')
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
                        default='./model/haarcascade_frontalface_model.xml',
                        help='Model file for face detection')
    parser.add_argument('--output',
                        type=str,
                        default="output.png",
                        help='Output image name')
    arguments = parser.parse_args()

    assert os.path.isfile(arguments.input_image), 'Input image data'

    demo_from_image(arguments)
