# nationality-recognition
## Install Packages
```
poetry install
```

## Demo
### Demo using the facial image
```
poetry run python demo/demo_from_image.py --input_image=path/to/image
```

### Demo using the video
```
poetry run python demo/demo_from_video.py --input_video=path/to/video
```

### Demo using the camera (Real-time demo)
```
poetry run python demo/demo_from_video.py --use_camera
```

## Train
```
poetry run python train.py --train_data_directory=path/to/train_data_directory --validation_data_directory=path/to/validation_data_directory
```

## Test
```
poetry run python test.py --test_data_directory=path/to/test_data_directory
```
