> Do you wanna have a bad time?

# detection-algorithms
Deepfake detection algorithm zoo.

## Project Setup

### Dataset

#### DFDC
Download [data from DFDC](https://www.kaggle.com/c/deepfake-detection-challenge) to `dataset` directory.

```shell
$ mkdir -p input/deepfake-detection-challenge
$ cd input/deepfake-detection-challenge
$ kaggle competitions download deepfake-detection-challenge
$ unzip deepfake-detection-challenge.zip
$ rm deepfake-detection-challenge.zip # unless you want to waste ~4GB disk space
```

#### FaceForensics
Refer to the [download instructions](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md) by FaceForensics.

```shell
$ mkdir -p input/faceforensics
$ cd input/faceforensics
$ python $(faceforensics-download-script) .
```

Replace `$(faceforensics-download-script)` to the actual path of the script.

### Dependencies
Install dependencies with this `pip` command:

```shell
$ pip3 install dlib jupyterlab numpy opencv-python pandas
```

Some dependencies, like `dlib` require build tools.
