###########
# Imports #
###########

""" Global """
import os
import argparse
from glob import glob

""" Local """
import constants
from Classifiers.BaselineClassifier import BaselineClassifier
from Classifiers.CustomClassifier import CustomClassifier
from Classifiers.DeepClassifier import DeepClassifier
from Classifiers.BOWClassifier import BOWClassifier

#############
# Functions #
#############

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for running classifier")
    parser.add_argument("-ci", "--catalog_images_folder", dest="catalog_images_folder", help="Path to catalog images folder", default=constants.CATALOG_IMAGES_PATH)
    parser.add_argument("-qi", "--query_path", dest="query_path", help="Path to query images", default=constants.CLASSIFICATION_QUERY_IMAGES_PATH)
    parser.add_argument("-gt", "--ground_truth_path", dest="ground_truth_path", help="Path to ground truth annotation", default=constants.CLASSIFICATION_GROUND_TRUTH_PATH)
    parser.add_argument("-clf", "--classifier", dest="classifier", help="Classifier : Baseline, Custom (default), BOW, Deep", default="Custom")
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
        return CustomClassifier(args.catalog_images_paths)

########
# Main #
########

if __name__ == "__main__":
    if not os.path.exists("./files"): os.makedirs("./files")
    args = parse_args()
    classifier = get_classifier(args)
    if args.one_query:
        scores = classifier.predict_query(args.query_path)
        top5labels = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:5]
        print("Top 5 labels :")
        for k in range(5):
            print("{}. Label = {} | Score = {}".format(k+1, top5labels[k], scores[top5labels[k]]))
    else:
        top1, top3, top5 = classifier.compute_metrics(args.query_images_paths, args.ground_truth_path)
        print("Metrics : Top1 = {:.3f}% | Top3 = {:.3f}% | Top5 = {:.3f}%".format(top1 * 100, top3 * 100, top5 * 100))