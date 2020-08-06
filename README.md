> Do you wanna have a bad time?

# detection-algorithms
Deepfake detection algorithm zoo.

## Project Setup

### Dataset
Download [data from DFDC](https://www.kaggle.com/c/deepfake-detection-challenge) to `dataset` directory.

```shell
$ mkdir -p input/deepfake-detection-challenge
$ cd input/deepfake-detection-challenge
$ kaggle competitions download deepfake-detection-challenge
$ unzip deepfake-detection-challenge.zip
$ rm deepfake-detection-challenge.zip # unless you want to waste ~4GB disk space
```

### Dependencies
Install these dependencies:
