###########
# Imports #
###########

""" Global """
import json
import numpy as np
from tqdm import tqdm

""" Local """
import constants

##############
# Classifier #
##############

class BaselineClassifier(object):
    def __init__(self, catalog_images_paths, params={}):
        self.catalog_images_paths = sorted(catalog_images_paths)
        self.get_params(params)
    
    def get_params(self, params):
        # <-- General --> #
        self.verbose = params.get("verbose", constants.VERBOSE)
        self.feature_extractor = params.get("feature_extractor", constants.DEFAULT_FEATURE_EXTRACTOR)
        self.matcher_index_params = params.get("matcher_index_params", constants.DEFAULT_MATCHER_INDEX_PARAMS)
        self.matcher_query_params = params.get("matcher_query_params", constants.DEFAULT_MATCHER_QUERY_PARAMS)
        self.background_label = params.get("background_label", constants.BACKGROUND_LABEL)
        self.image_size = params.get("image_size", constants.DEFAULT_CLASSIFIER_IMAGE_SIZE)
        self.keypoint_stride = params.get("keypoint_stride", constants.DEFAULT_CLASSIFIER_KEYPOINT_STRIDE)
        self.keypoint_sizes = params.get("keypoint_sizes", constants.DEFAULT_CLASSIFIER_KEYPOINT_SIZES)
        
        # <-- Classifier Custom --> #
        self.matcher_path_custom = params.get("matcher_path", constants.DEFAULT_CLASSIFIER_CUSTOM_MATCHER_PATH)
        self.force_matcher_compute = params.get("force_matcher_compute", constants.DEFAULT_CLASSIFIER_CUSTOM_FORCE_MATCHER_COMPUTE)
        self.k_nn_custom = params.get("k_nn", constants.DEFAULT_CLASSIFIER_CUSTOM_K_NN)
        self.score_sigma = params.get("sigma", constants.DEFAULT_CLASSIFIER_CUSTOM_SCORE_SIGMA)

        # <-- Classifier Deep --> #
        self.matcher_path_deep = params.get("matcher_path", constants.DEFAULT_CLASSIFIER_DEEP_MATCHER_PATH)
        self.k_nn_deep = params.get("k_nn", constants.DEFAULT_CLASSIFIER_DEEP_K_NN)
        self.triplet_margin = params.get("triplet_margin", constants.DEFAULT_CLASSIFIER_DEEP_TRIPLET_MARGIN)
        self.n_train_epochs = params.get("n_train_epochs", constants.DEFAULT_CLASSIFIER_DEEP_TRAIN_EPOCHS)
        self.batch_size = params.get("batch_size", constants.DEFAULT_CLASSIFIER_DEEP_TRAIN_BATCH_SIZE)
        self.augment_factor = params.get("augment_factor", constants.DEFAULT_CLASSIFIER_DEEP_TRAIN_AUGMENT_FACTOR)
        self.model_path = params.get("model_path", constants.DEFAULT_CLASSIFIER_DEEP_MODEL_PATH)
        self.force_train = params.get("force_train", constants.DEFAULT_CLASSIFIER_DEEP_FORCE_TRAIN)
        
        # <-- Classifier BoW --> #
        self.vocab_size = params.get("vocab_size", constants.DEFAULT_CLASSIFIER_BOW_VOCAB_SIZE)
        self.vocab_path = params.get("vocab_path", constants.DEFAULT_CLASSIFIER_BOW_VOCAB_PATH)
        self.catalog_features_path = params.get("catalog_features_path", constants.DEFAULT_CLASSIFIER_BOW_CATALOG_FEATURES_PATH)
        self.force_vocab_compute = params.get("force_vocab_compute", constants.DEFAULT_CLASSIFIER_BOW_FORCE_VOCAB_COMPUTE)
        self.force_catalog_features_compute = params.get("force_catalog_features_compute", constants.DEFAULT_CLASSIFIER_BOW_FORCE_CATALOG_FEATURES_COMPUTE)

    def predict_query(self, query):
        return {}
    
    def predict_query_batch(self, query_paths):
        iterator = query_paths
        if self.verbose: iterator = tqdm(iterator, desc="Query prediction")

        results = {}
        for query_path in iterator:
            query_id = query_path.split("/")[-1]
            results[query_id] = self.predict_query(query_path)
        
        return results

    def compute_top_k_accuracy(self, ground_truth, predictions, k):
        nb_correct = counter = 0
        for img_id in ground_truth:
            if ground_truth[img_id] != self.background_label and img_id in predictions:
                predicted_labels = sorted(predictions[img_id].keys(), key=lambda x: predictions[img_id][x], reverse=True)[:k]
                if ground_truth[img_id] in predicted_labels:
                    nb_correct += 1
                counter += 1
        if counter == 0:
            return 0.0
        return 1. * nb_correct / counter
    
    def compute_metrics(self, query_paths, ground_truth_path):
        with open(ground_truth_path, "r") as f: ground_truth = json.load(f)
        predictions = self.predict_query_batch(query_paths)
        top1 = self.compute_top_k_accuracy(ground_truth, predictions, 1)
        top3 = self.compute_top_k_accuracy(ground_truth, predictions, 3)
        top5 = self.compute_top_k_accuracy(ground_truth, predictions, 5)
        return top1, top3, top5