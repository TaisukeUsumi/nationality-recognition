import argparse
import json
import os

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


def test(arguments):
    test_data_generator = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_data_generator.flow_from_directory(
        arguments.test_data_directory,
        target_size=(arguments.image_size, arguments.image_size),
        batch_size=arguments.batch_size,
        classes=arguments.classes,
        color_mode='rgb',
        class_mode='categorical')

    test_num = test_generator.samples

    model = load_model(arguments.nationality_recognition_model)
    loss, accuracy = model.evaluate_generator(test_generator,
                                              steps=test_num //
                                              arguments.batch_size)

    result = {"loss": loss, "accuracy": accuracy}

    with open(arguments.output, 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--test_data_directory',
                        type=str,
                        default='',
                        help='Input test data directory')
    parser.add_argument('--image_size',
                        type=int,
                        default=128,
                        help='Model input image size')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size')
    parser.add_argument('--classes',
                        type=str,
                        default=['chinese', 'japanese', 'korean'],
                        help='Target classes to recognize')
    parser.add_argument('--nationality_recognition_model',
                        type=str,
                        default='./model/nationality_recognition_model.h5',
                        help='Model file for nationality recognition')
    parser.add_argument('--output',
                        type=str,
                        default="output.json",
                        help='Output file name')
    arguments = parser.parse_args()

    assert os.path.isdir(
        arguments.test_data_directory), 'Input test data directory'

    test(arguments)
