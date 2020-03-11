###########
# Imports #
###########

""" Global """
import os
import cv2
import nmslib
import numpy as np
from tqdm import tqdm

""" Local """
from Classifiers.BaselineClassifier import BaselineClassifier
import utils

##############
# Classifier #
##############

class CustomClassifier(BaselineClassifier):
    def __init__(self, catalog_images_paths, params={}):
        super(CustomClassifier, self).__init__(catalog_images_paths, params=params)
        self.config_matcher()

    def config_matcher(self):
        self.matcher = nmslib.init(method="hnsw", space="l2")
        if not self.force_matcher_compute and os.path.exists(self.matcher_path_custom):
            if self.verbose: print("Loading index...")
            self.matcher.loadIndex(self.matcher_path_custom)
            if self.verbose: print("Index loaded !")
        else:
            self.get_catalog_descriptors()
            if self.verbose: print("Creating index...")
            self.matcher.addDataPointBatch(self.catalog_descriptors)
            self.matcher.createIndex(self.matcher_index_params, print_progress=self.verbose)
            self.matcher.setQueryTimeParams(self.matcher_query_params)
            if self.verbose: print("Index created !")

            if self.verbose: print("Saving index...")
            self.matcher.saveIndex(self.matcher_path_custom)
            if self.verbose: print("Index saved !")

    def get_catalog_descriptors(self):
        iterator = self.catalog_images_paths
        if self.verbose: iterator = tqdm(iterator, desc="Catalog description")

        self.catalog_descriptors = []
        for path in iterator:
            img = utils.read_image(path, size=self.image_size)
            keypoints = utils.get_keypoints(img, self.keypoint_stride, self.keypoint_sizes)
            descriptors = utils.get_descriptors(img, keypoints, self.feature_extractor)
            self.catalog_descriptors.append(descriptors)

        self.catalog_descriptors = np.array(self.catalog_descriptors)
        self.catalog_descriptors = self.catalog_descriptors.reshape(-1, self.catalog_descriptors.shape[-1])

    def get_query_scores(self, query_descriptors):
        matches = self.matcher.knnQueryBatch(query_descriptors, k=self.k_nn_custom)
        trainIdx = np.array([m[0] for m in matches])
        distances = np.array([m[1] for m in matches])
        scores_matrix = np.exp(-(distances / self.score_sigma) ** 2)
        scores = {}
        for ind, nn_trainIdx in enumerate(trainIdx):
            for k, idx in enumerate(nn_trainIdx):
                catalog_path = self.catalog_images_paths[idx // query_descriptors.shape[0]]
                catalog_label = catalog_path.split("/")[-1][:-4]
                scores[catalog_label] = scores.get(catalog_label, 0) + scores_matrix[ind, k]
        return scores

    def predict_query(self, query, score_threshold=None):
        if type(query) in [str, np.string_]: query_img = utils.read_image(query, size=self.image_size)
        else: query_img = cv2.resize(query, (self.image_size, self.image_size))
        query_keypoints = utils.get_keypoints(query_img, self.keypoint_stride, self.keypoint_sizes)
        query_descriptors = utils.get_descriptors(query_img, query_keypoints, self.feature_extractor)
        scores = self.get_query_scores(query_descriptors)
        return scores