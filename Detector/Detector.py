###########
# Imports #
###########

""" Global """
import cv2
import numpy as np
import nmslib
from tqdm import tqdm
from glob import glob
import os
import json
from sklearn.cluster import DBSCAN
import xml.etree.cElementTree as ET
from easydict import EasyDict as edict

""" Local """
import constants
import utils

############
# Detector #
############

class Detector(object):
    def __init__(self, catalog_images_paths, params={}):
        self.catalog_images_paths = sorted(catalog_images_paths)
        self.get_params(params)
        self.config_matcher()

    def get_params(self, params):
        self.feature_extractor = params.get("feature_extractor", constants.DEFAULT_FEATURE_EXTRACTOR)
        self.catalog_image_widths = params.get("catalog_image_widths", constants.DEFAULT_DETECTOR_CATALOG_IMAGE_WIDTHS)
        self.query_image_width = params.get("query_image_width", constants.DEFAULT_DETECTOR_QUERY_IMAGE_WIDTH)
        self.catalog_keypoint_stride = params.get("catalog_keypoint_stride", constants.DEFAULT_DETECTOR_CATALOG_KEYPOINT_STRIDE)
        self.query_keypoint_stride = params.get("query_keypoint_stride", constants.DEFAULT_DETECTOR_QUERY_KEYPOINT_STRIDE)
        self.keypoint_sizes = params.get("keypoint_sizes", constants.DEFAULT_DETECTOR_KEYPOINT_SIZES)
        self.matcher_index_params = params.get("matcher_index_params", constants.DEFAULT_MATCHER_INDEX_PARAMS)
        self.matcher_query_params = params.get("matcher_query_params", constants.DEFAULT_MATCHER_QUERY_PARAMS)
        self.matcher_path = params.get("matcher_path", constants.DEFAULT_DETECTOR_MATCHER_PATH)
        self.force_matcher_compute = params.get("force_matcher_compute", constants.DEFAULT_DETECTOR_FORCE_MATCHER_COMPUTE)
        self.bbox_threshold = params.get("bbox_threshold", constants.DEFAULT_DETECTOR_BBOX_THRESHOLD)
        self.k_nn = params.get("k_nn", constants.DEFAULT_DETECTOR_K_NN)
        self.score_sigma = params.get("sigma", constants.DEFAULT_DETECTOR_SCORE_SIGMA)
        self.verbose = params.get("verbose", constants.VERBOSE)

    def config_matcher(self):
        self.matcher = nmslib.init(method="hnsw", space="l2")
        if not self.force_matcher_compute and os.path.exists(self.matcher_path):
            self.get_catalog_data(compute_descriptors=False)
            if self.verbose: print("Loading index...")
            self.matcher.loadIndex(self.matcher_path)
            if self.verbose: print("Index loaded !")
        else:
            self.get_catalog_data()
            if self.verbose: print("Creating index...")
            self.matcher.addDataPointBatch(self.catalog_data["descriptors"])
            self.matcher.createIndex(self.matcher_index_params, print_progress=self.verbose)
            self.matcher.setQueryTimeParams(self.matcher_query_params)
            if self.verbose: print("Index created !")
            
            if self.verbose: print("Saving index...")
            self.matcher.saveIndex(self.matcher_path)
            if self.verbose: print("Index saved !")

    def get_catalog_data(self, compute_descriptors=True):
        iterator = self.catalog_images_paths
        if self.verbose: iterator = tqdm(iterator, desc="Get catalog data")
        
        self.catalog_data = {
            "keypoints": [],
            "descriptors": [],
            "labels": [],
            "shapes": [],
        }
        for catalog_path in iterator:
            for width in self.catalog_image_widths:
                img = utils.read_image(catalog_path, width=width)
                label = catalog_path.split("/")[-1][:-4]
                keypoints = utils.get_keypoints(img, self.catalog_keypoint_stride, self.keypoint_sizes)
                self.catalog_data["keypoints"] += list(keypoints)
                self.catalog_data["labels"] += [label] * len(keypoints)
                self.catalog_data["shapes"] += [img.shape[:2]] * len(keypoints)
                if compute_descriptors:
                    descriptors = utils.get_descriptors(img, keypoints, self.feature_extractor)
                    self.catalog_data["descriptors"] += list(descriptors)
        
        self.catalog_data["descriptors"] = np.array(self.catalog_data["descriptors"])

    def get_matches_results(self, query_kpts_data, query_descriptors, query_shape):
        # Matching
        if self.verbose: print("Query matching...")
        matches = self.matcher.knnQueryBatch(query_descriptors, k=self.k_nn)
        
        # Result
        trainIds = np.array([m[0] for m in matches])
        distances = np.array([m[1] for m in matches])
        scores = np.exp(-(distances / self.score_sigma) ** 2)

        # Updating keypoints
        for i, kpt in enumerate(query_kpts_data):
            label_scores = {}
            for k in range(self.k_nn):
                label = self.catalog_data["labels"][trainIds[i, k]]
                label_scores[label] = label_scores.get(label, 0) + scores[i, k]
            kpt.label = sorted(label_scores.keys(), key=lambda label: label_scores[label], reverse=True)[0]
            kpt.score = label_scores[kpt.label]
            kpt.query_shape = np.array(query_shape[:2])
            kpt.catalog_pt = np.array(self.catalog_data["keypoints"][trainIds[i, 0]].pt)
            kpt.catalog_shape = np.array(self.catalog_data["shapes"][trainIds[i, 0]][:2])

    def get_raw_bboxes(self, query_kpts_data):
        iterator = query_kpts_data
        if self.verbose: iterator = tqdm(iterator, desc="Raw bboxes")

        bboxes = {}
        for kpt in iterator:
            if kpt.score > 0.9:
                query_coord = kpt.query_pt - kpt.query_shape / 2.
                catalog_coord = kpt.catalog_pt - kpt.catalog_shape / 2.
                catalog_center = np.array([0, 0])
                query_center = query_coord + (catalog_center - catalog_coord)
                bbox = edict({
                    "kpt": kpt, "score": kpt.score,
                    "feature": query_center,
                })
                if kpt.label in bboxes: bboxes[kpt.label].append(bbox)
                else: bboxes[kpt.label] = [bbox]
        return bboxes

    def filter_bboxes(self, bboxes, query_shape):
        iterator = bboxes
        if self.verbose: iterator = tqdm(iterator, desc="Filtering bboxes")
        
        filtered_bboxes = {}
        for label in iterator:
            label_bboxes = np.array(bboxes[label])
            bbox_features = np.array([bbox.feature for bbox in label_bboxes])
            clusters = DBSCAN(eps=10, min_samples=3).fit_predict(bbox_features)
            for k in set(clusters):
                if k != -1:
                    keypoints = np.array([bbox.kpt for bbox in label_bboxes[clusters == k]])
                    bbox = utils.find_bbox_from_keypoints(keypoints)
                    if bbox is not None:
                        if label in filtered_bboxes: filtered_bboxes[label].append(bbox)
                        else: filtered_bboxes[label] = [bbox]
            filtered_bboxes[label] = utils.apply_custom_nms(filtered_bboxes.get(label, []))
        
        return filtered_bboxes

    def merge_bboxes(self, bboxes, query_shape):
        iterator = bboxes
        if self.verbose: iterator = tqdm(iterator, desc="Merging bboxes")

        merged_bboxes = []
        for label in iterator:
            for bbox in bboxes[label]:
                bbox.label = label
                merged_bboxes.append(bbox)
        merged_bboxes = np.array(merged_bboxes)
        
        filtered_merged_bboxes = []
        bbox_coords = np.array([bbox.coords for bbox in merged_bboxes])
        clusters = DBSCAN(eps=20, min_samples=1).fit_predict(bbox_coords)
        for k in set(clusters):
            scores = np.array([bbox.score for bbox in merged_bboxes[clusters == k]])
            labels = np.array([bbox.label for bbox in merged_bboxes[clusters == k]])
            coords = np.array([bbox.coords for bbox in merged_bboxes[clusters == k]])
            label_scores = {
                label: np.sum(scores[labels == label])
                for label in set(list(labels))
            }
            best_label = sorted(label_scores.keys(), key=lambda label: label_scores[label], reverse=True)[0]
            filtered_merged_bboxes.append(edict({
                "label": best_label,
                "score": label_scores[best_label],
                "coords": np.mean(coords[labels == best_label], axis=0),
            }))
        
        filtered_merged_bboxes = utils.apply_custom_nms(filtered_merged_bboxes)

        for bbox in filtered_merged_bboxes:
            xmin, ymin, xmax, ymax = bbox.coords
            xmin = max(xmin, 0); ymin = max(ymin, 0)
            xmax = min(xmax, query_shape[1] - 1)
            ymax = min(ymax, query_shape[0] - 1)
            bbox.coords = np.array([xmin, ymin, xmax, ymax]).astype(int)
        
        return filtered_merged_bboxes

    def add_classifier_score(self, bboxes, query_img, classifier):
        iterator = bboxes
        if self.verbose: iterator = tqdm(iterator, desc="Classifier score")

        for bbox in iterator:
            xmin, ymin, xmax, ymax = bbox.coords
            if xmin < xmax and ymin < ymax:
                crop_img = query_img[ymin:ymax, xmin:xmax]
                classifier_scores = classifier.predict_query(crop_img)
                best_label = sorted(classifier_scores.keys(), key=lambda x: classifier_scores[x], reverse=True)[0]
                bbox.score = 2. / (1. / classifier_scores[best_label] + 1. / bbox.score)
                bbox.label = best_label
            else:
                bbox.score = 0

        return bboxes

    def filter_bboxes_with_threshold(self, bboxes):
        return [bbox for bbox in bboxes if bbox.score > self.bbox_threshold]

    def reshape_bboxes_original_size(self, bboxes, original_size, current_size):
        original_h, original_w = original_size
        current_h, current_w = current_size
        for bbox in bboxes:
            xmin, xmax, ymin, ymax = bbox.coords
            xmin, xmax = list(map(lambda x: 1. * x * original_w / current_w, (xmin, xmax)))
            ymin, ymax = list(map(lambda y: 1. * y * original_h / current_h, (ymin, ymax)))
            bbox.coords = np.array([xmin, xmax, ymin, ymax]).astype(np.int)
        return bboxes

    def predict_query(self, query_path, classifier=None, apply_threshold=True):
        # Read img
        query_img = utils.read_image(query_path, width=self.query_image_width)
        query_original_h, query_original_w = cv2.imread(query_path).shape[:2]

        # Get keypoints
        query_keypoints = utils.get_keypoints(query_img, self.query_keypoint_stride, self.keypoint_sizes)
        query_kpts_data = np.array([utils.keypoint2data(kpt) for kpt in query_keypoints])
        
        # Get descriptors
        if self.verbose: print("Query description...")
        query_descriptors = utils.get_descriptors(query_img, query_keypoints, self.feature_extractor)

        # Matching
        self.get_matches_results(query_kpts_data, query_descriptors, query_img.shape)

        # Get bboxes
        bboxes = self.get_raw_bboxes(query_kpts_data)
        bboxes = self.filter_bboxes(bboxes, query_img.shape)
        bboxes = self.merge_bboxes(bboxes, query_img.shape)
        if classifier is not None:
            bboxes = self.add_classifier_score(bboxes, query_img, classifier)
        if apply_threshold:
            bboxes = self.filter_bboxes_with_threshold(bboxes)
        bboxes = self.reshape_bboxes_original_size(bboxes, (query_original_h, query_original_w), query_img.shape[:2])

        return bboxes

    def draw_bboxes(self, query_path, bboxes, output_path):
        query_img = utils.read_image(query_path)
        boxes_img = utils.draw_bboxes(query_img, bboxes)
        cv2.imwrite(output_path, boxes_img)

    def get_xml(self, query_filename, bboxes, out_folder):
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = query_filename
        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox.coords
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = bbox.label
            ET.SubElement(obj, "confidence").text = str(bbox.score)
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "ymax").text = str(ymax)
        tree = ET.ElementTree(root)
        tree.write(out_folder + "/" + ".".join(query_filename.split(".")[:-1]) + ".xml")

    def predict_query_batch(self, query_paths, classifier=None, out_folder=None):
        results = {}
        for k, query_path in enumerate(query_paths):
            query_id = query_path.split("/")[-1]
            if self.verbose:
                print("\n")
                print("="*40)
                print(query_id + "... (" + str(k + 1) + "/" + str(len(query_paths)) + ")")
                print("="*40)
            results[query_id] = self.predict_query(query_path, classifier=classifier, apply_threshold=False)
            
            if out_folder is not None:
                self.get_xml(query_id, results[query_id], out_folder)
        
        return results
