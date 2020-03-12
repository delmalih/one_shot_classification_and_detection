# One-Shot-Detector

## Requirements

- Anaconda 3 ([install. instructions](https://www.anaconda.com/distribution/))

## Installation steps

Start by cloning this repo and mv into it:

```
https://github.com/delmalih/one_shot_classification_and_detection
mv one_shot_classification_and_detection
```

Then, create a conda environment and install all required packages:

```
conda create -n OneShotClassifierDetector python=3.6
conda activate OneShotDetector
pip install -r requirements.txt
```

You're all set :)

## How to use it ?

Feel free to edit the `constants.py` file to tune the parameters.

### Classifiers

Command to run the classifier :

```
python run_classifier.py -ci <CATALOG_IMAGES_FOLDER> \ # Required
                         -qi <QUERY_PATH> \ # Required (could be a single image or a folder)
                         -gt <GROUND_TRUTH_PATH> \ # In case you want to compute the accuracy
                         -clf <CLASSIFIER> \ # Type of Classifier : Baseline, Custom (default), BOW, Deep
                         --one_query \ # To be set if you want to predict only one query image
```

### Detector

Command to run the classifier :

```
python run_detector.py -ci <CATALOG_IMAGES_FOLDER> \ # Required
                       -qi <QUERY_PATH> \ # Required (could be a single image or a folder)
                       -o <OUTPUT_PATH> \ # Required
                       -clf <CLASSIFIER> \ # Choose a classifier to improve the results (or not) : Baseline, Custom, BOW, Deep or None (default)
                       --one_query \ # To be set if you want to predict only one query image
```