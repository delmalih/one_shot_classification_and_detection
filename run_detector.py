###########
# Imports #
###########

""" Global """
import os
import argparse
from glob import glob

""" Local """
import constants
from Detector.Detector import Detector
from Classifiers.BaselineClassifier import BaselineClassifier
from Classifiers.CustomClassifier import CustomClassifier
from Classifiers.DeepClassifier import DeepClassifier
from Classifiers.BOWClassifier import BOWClassifier

#############
# Functions #
#############

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for running detector")
    parser.add_argument("-ci", "--catalog_images_folder", dest="catalog_images_folder", help="Path to catalog images folder", default=constants.CATALOG_IMAGES_PATH)
    parser.add_argument("-qi", "--query_path", dest="query_path", help="Path to query image", default=constants.DETECTOR_QUERY_IMAGES_PATH)
    parser.add_argument("-o", "--output_path", dest="output_path", help="Path for the output", required=True)
    parser.add_argument("-clf", "--classifier", dest="classifier", help="Classifier : Baseline, Custom, BOW, Deep")
    parser.add_argument("--one_query", action="store_true", help="Predict only one query")
    
    args = parser.parse_args()
    args.catalog_images_paths = glob(args.catalog_images_folder + "/*")
    if not args.one_query:
        args.query_images_paths = glob(args.query_path + "/*")
    
    return args

def get_classifier(args):
    if args.classifier == "Baseline":
        return BaselineClassifier(args.catalog_images_paths)
    elif args.classifier == "Custom":
        return CustomClassifier(args.catalog_images_paths)
    elif args.classifier == "Deep":
        return DeepClassifier(args.catalog_images_paths)
    elif args.classifier == "BOW":
        return BOWClassifier(args.catalog_images_paths)
    else:
        return None

########
# Main #
########

if __name__ == "__main__":
    if not os.path.exists("./files"): os.makedirs("./files")
    args = parse_args()
    detector = Detector(args.catalog_images_paths)
    classifier = get_classifier(args)
    if args.one_query:
        bboxes = detector.predict_query(args.query_path, classifier=classifier)
        detector.draw_bboxes(args.query_path, bboxes, args.output_path)
    else:
        detector.predict_query_batch(args.query_images_paths, classifier=classifier, out_folder=args.output_path)