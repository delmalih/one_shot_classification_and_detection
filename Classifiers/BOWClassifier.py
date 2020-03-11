###########
# Imports #
###########

""" Global """
import os
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
import sklearn.metrics.pairwise as sklearn_pairwise

""" Local """
from Classifiers.BaselineClassifier import BaselineClassifier
import utils

##############
# Classifier #
##############

class BOWClassifier(BaselineClassifier):
    def __init__(self, catalog_images_paths, params={}):
        super(BOWClassifier, self).__init__(catalog_images_paths, params=params)
        self.build_vocab()
        self.build_catalog_features()
    
    def build_vocab(self):
        if not self.force_vocab_compute and os.path.exists(self.vocab_path):
            if self.verbose: print("Loading vocab...")
            with open(self.vocab_path, "rb") as f:
                self.vocab = pickle.load(f)
            if self.verbose: print("Vocab loaded !")
        else:
            iterator = self.catalog_images_paths
            if self.verbose: iterator = tqdm(iterator, desc="Vocab construction")
            descriptors = []
            image_ids = []
            for i, image_path in enumerate(iterator):
                image = utils.read_image(image_path, size=self.image_size)
                keypoints = utils.get_keypoints(image, self.keypoint_stride, self.keypoint_sizes)
                desc = utils.get_descriptors(image, keypoints, self.feature_extractor)
                descriptors += list(desc)
                image_ids += [i for _ in range(len(keypoints))]
            descriptors = np.array(descriptors)
            image_ids = np.array(image_ids)

            if self.verbose: print("KMeans step...")
            kmeans = MiniBatchKMeans(n_clusters=self.vocab_size, init_size=3 * self.vocab_size)
            clusters = kmeans.fit_predict(descriptors)
            
            if self.verbose: print("Computing Idfs...")
            self.vocab = {}
            self.vocab["features"] = kmeans.cluster_centers_
            self.vocab["idf"] = np.zeros((self.vocab["features"].shape[0],))
            nb_documents = len(self.catalog_images_paths)
            for cluster in set(clusters):
                nb_documents_containing_cluster = len(set(image_ids[clusters == cluster]))
                self.vocab["idf"][cluster] = np.log(1. * nb_documents / nb_documents_containing_cluster)

            if self.verbose: print("Saving vocal...")
            with open(self.vocab_path, "wb") as f:
                pickle.dump(self.vocab, f) 
            if self.verbose: print("Vocab saved !")

    def build_catalog_features(self):
        if not self.force_vocab_compute and not self.force_catalog_features_compute and os.path.exists(self.catalog_features_path):
            if self.verbose: print("Loading catalog features...")
            with open(self.catalog_features_path, "rb") as f:
                self.catalog_features = pickle.load(f)
            if self.verbose: print("Catalog features loaded !")
        else:
            iterator = self.catalog_images_paths
            if self.verbose: iterator = tqdm(iterator, desc="Computing catalog features")
            self.catalog_features = []
            for image_path in iterator:
                image = utils.read_image(image_path, size=self.image_size)
                features = self.compute_image_features(image)
                self.catalog_features.append(features)
            self.catalog_features = np.array(self.catalog_features)

            if self.verbose: print("Saving catalog features...")
            with open(self.catalog_features_path, "wb") as f:
                pickle.dump(self.catalog_features, f) 
            if self.verbose: print("Catalog features saved !")

    def compute_image_features(self, image):
        keypoints = utils.get_keypoints(image, self.keypoint_stride, self.keypoint_sizes)
        descriptors = utils.get_descriptors(image, keypoints, self.feature_extractor)
        distances = sklearn_pairwise.pairwise_distances(descriptors, self.vocab["features"], metric="cosine")
        softmax_distances = np.exp(1. - distances) / np.sum(np.exp(1. - distances), axis=1, keepdims=True)
        features = 1. * np.sum(softmax_distances, axis=0) / len(softmax_distances) * self.vocab["idf"]
        return features

    def match_query(self, query_features):
        distances = sklearn_pairwise.pairwise_distances(np.array([query_features]), self.catalog_features, metric="cosine")[0]
        scores = {}
        for k, catalog_path in enumerate(self.catalog_images_paths):
            label = catalog_path.split("/")[-1][:-4] 
            scores[label] = 1. - distances[k]
        return scores

    def predict_query(self, query):
        if type(query) in [str, np.string_]: query_img = utils.read_image(query, size=self.image_size)
        else: query_img = cv2.resize(query, (self.image_size, self.image_size))
        query_features = self.compute_image_features(query_img)
        scores = self.match_query(query_features)
        return scores
