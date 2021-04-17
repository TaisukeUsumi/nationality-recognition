import argparse
import os
import pickle

from keras.applications import MobileNetV2
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.layers import Conv2D, Dense, Dropout, Flatten, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


def step_decay(epoch):
    x = 1e-3
    if epoch >= 60:
        x = 1e-4
    if epoch >= 100:
        x = 1e-5
    return x


def net(image_size, pretrained_model, alpha=1.0):
    inputs = Input(shape=(image_size, image_size, 3))
    model_mobilenet = MobileNetV2(input_shape=(image_size, image_size, 3),
                                  alpha=alpha,
                                  include_top=False,
                                  weights=pretrained_model,
                                  input_tensor=None,
                                  pooling=None)
    x = model_mobilenet(inputs)
    conv_1 = Conv2D(128, (1, 1), activation='relu')(x)
    flat_1 = Flatten()(conv_1)
    drop_1 = Dropout(0.5)(flat_1)
    dence_1 = Dense(128, activation='relu', name='feat_a')(drop_1)
    dence_2 = Dense(32, activation='relu', name='feat_b')(dence_1)
    outputs = Dense(3, activation="softmax")(dence_2)
    model = Model(inputs=inputs, outputs=outputs)

    return model


def train(arguments):
    if not os.path.exists(arguments.model_output_directory):
        os.makedirs(arguments.model_output_directory)

    train_data_generator = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=arguments.horizontal_flip,
        rotation_range=arguments.rotation_range,
        brightness_range=arguments.brightness_range)

    validation_data_generator = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_data_generator.flow_from_directory(
        arguments.train_data_directory,
        target_size=(arguments.image_size, arguments.image_size),
        batch_size=arguments.batch_size,
        classes=arguments.classes,
        color_mode='rgb',
        class_mode='categorical')

    validation_generator = validation_data_generator.flow_from_directory(
        arguments.validation_data_directory,
        target_size=(arguments.image_size, arguments.image_size),
        batch_size=arguments.batch_size,
        classes=arguments.classes,
        color_mode='rgb',
        class_mode='categorical')

    train_num = train_generator.samples
    validation_num = validation_generator.samples

    model = net(arguments.image_size, arguments.pretrained_model)

    model.summary()

    model.compile(optimizer=Adam(lr=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    lr_decay = LearningRateScheduler(step_decay)

    callbacks = \
        [ModelCheckpoint(arguments.model_output_directory + '/weights.{epoch:02d}-{val_accuracy:.2f}-{val_loss:.2f}.h5',
                         monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto',
                         period=1), lr_decay]

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_num // arguments.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_num // arguments.batch_size,
        epochs=arguments.epochs,
        callbacks=callbacks,
        verbose=1,
        shuffle=True)

    model.save(arguments.model_output_directory + '/final_model.h5')

    with open(arguments.model_output_directory + '/learning_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train face network')
    parser.add_argument('--train_data_directory',
                        type=str,
                        default='',
                        help='Input train data directory')
    parser.add_argument('--validation_data_directory',
                        type=str,
                        default='',
                        help='Input validation data directory')
    parser.add_argument('--image_size',
                        type=int,
                        default=128,
                        help='Model input image size')
    parser.add_argument('--horizontal_flip',
                        action='store_true',
                        help='Horizontal flip')
    parser.add_argument('--rotation_range',
                        type=int,
                        default=30,
                        help='Rotation range')
    parser.add_argument('--brightness_range',
                        type=float,
                        default=[0.6, 1.4],
                        help='Brightness range')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='Batch size')
    parser.add_argument('--classes',
                        type=str,
                        default=['chinese', 'japanese', 'korean'],
                        help='Target classes to recognize')
    parser.add_argument('--pretrained_model',
                        type=str,
                        default=None,
                        help='Pretrained model')
    parser.add_argument('--epochs', type=int, default=120, help='Epochs')
    parser.add_argument('--model_output_directory',
                        type=str,
                        default="output",
                        help='Name of the directory to output models')
    arguments = parser.parse_args()

    assert os.path.isdir(
        arguments.train_data_directory), 'Input train data directory'
    assert os.path.isdir(
        arguments.validation_data_directory), 'Input validation data directory'

    train(arguments)
