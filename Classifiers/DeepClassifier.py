###########
# Imports #
###########

""" Global """
import os
import cv2
import nmslib
import numpy as np
from tqdm import tqdm
from tensorflow import keras
import imgaug.augmenters as iaa

""" Local """
from Classifiers.BaselineClassifier import BaselineClassifier
import utils
from utils import triplet_loss

##############
# Classifier #
##############

class DeepClassifier(BaselineClassifier):
    def __init__(self, catalog_images_paths, params={}):
        super(DeepClassifier, self).__init__(catalog_images_paths, params=params)
        self.get_model()
        if self.force_train or not os.path.exists(self.model_path):
            if self.verbose: print("Training model ...")
            self.train()
        self.config_matcher()

    def config_matcher(self):
        self.matcher = nmslib.init(method="hnsw", space="l2")
        if not self.force_matcher_compute and os.path.exists(self.matcher_path_deep):
            if self.verbose: print("Loading index...")
            self.matcher.loadIndex(self.matcher_path_deep)
            if self.verbose: print("Index loaded !")
        else:
            self.get_catalog_descriptors()
            if self.verbose: print("Creating index...")
            self.matcher.addDataPointBatch(self.catalog_descriptors)
            self.matcher.createIndex(self.matcher_index_params, print_progress=self.verbose)
            self.matcher.setQueryTimeParams(self.matcher_query_params)
            if self.verbose: print("Index created !")

            if self.verbose: print("Saving index...")
            self.matcher.saveIndex(self.matcher_path_deep)
            if self.verbose: print("Index saved !")

    def get_model(self):
        def triplet_loss_ft(y_true, y_pred): return triplet_loss.batch_all_triplet_loss(y_true, y_pred, self.triplet_margin)[0]
        def triplet_metric_ft(y_true, y_pred): return triplet_loss.batch_all_triplet_loss(y_true, y_pred, self.triplet_margin)[1]
        input_layer = keras.layers.Input(shape=(self.image_size, self.image_size, 3))
        x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(input_layer)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPool2D()(x)
        x = keras.layers.Conv2D(512, 3, padding="same", activation="relu")(x)
        encoding = keras.layers.GlobalMaxPool2D()(x)
        triplet_output = keras.layers.Lambda(lambda x: keras.backend.l2_normalize(x, axis=-1))(encoding)
        self.model = keras.models.Model(input_layer, triplet_output)
        self.model.compile(optimizer="adam", loss=triplet_loss_ft, metrics=[triplet_metric_ft])
        if os.path.exists(self.model_path):
            try:
                if self.verbose: print("Loading weights ...")
                self.model.load_weights(self.model_path)
            except:
                if self.verbose: print("Failed to load model weights")

    def get_augmenter(self):
        seq = iaa.Sequential([
            iaa.SomeOf((2, 6), [
                iaa.Flipud(0.5),
                iaa.Rot90(k=(0, 3)),
                iaa.GaussianBlur(sigma=(0.0, 3.0)),
                iaa.MotionBlur(angle=(0, 360), k=(3, 8)),
                iaa.Add((-50, 5)),
                iaa.AddElementwise((-20, 2)),
                iaa.AdditiveGaussianNoise(scale=0.05*255),
                iaa.Multiply((0.3, 1.05)),
                iaa.SaltAndPepper(p=(0.1, 0.3)),
                iaa.JpegCompression(compression=(20, 90)),
                iaa.Affine(shear=(-15, 15)),
                iaa.Affine(rotate=(-10, 10)),
            ])
        ])
        return seq

    def get_batch(self, paths, augment=True):
        # Original
        original_images = [utils.read_image(path, size=self.image_size) for path in paths]
        original_labels = [self.catalog_images_paths.index(path) for path in paths]

        # Augmented
        if augment:
            augmenter = self.get_augmenter()
            total_images = [image for image in original_images for k in range(self.augment_factor)]
            total_labels = [label for label in original_labels for k in range(self.augment_factor)]
            total_images = augmenter(images=total_images)
        else:
            total_images = original_images
            total_labels = original_labels

        total_images = np.array(total_images)
        total_labels = np.array(total_labels)
        
        # Shuffle
        indexes = list(range(len(total_images)))
        np.random.shuffle(indexes)
        total_images = total_images[indexes]
        total_labels = total_labels[indexes]

        total_images = total_images[:, :, :, ::-1] / 255.0

        return total_images, total_labels

    def train(self):
        for epoch in range(self.n_train_epochs):
            paths = np.random.choice(self.catalog_images_paths, size=self.batch_size)
            batch_img, batch_labels = self.get_batch(paths)
            loss, metric = self.model.train_on_batch(batch_img, batch_labels)
            print("Epoch: {} | Loss = {:.4f} | Metric = {:.4f}%".format(epoch, loss, metric * 100))
            self.model.save_weights(self.model_path)

    def get_catalog_descriptors(self):
        iterator = range(0, len(self.catalog_images_paths), self.batch_size)
        if self.verbose: iterator = tqdm(iterator, desc="Catalog description")

        self.catalog_descriptors = []
        for i in iterator:
            batch_imgs, _ = self.get_batch(self.catalog_images_paths[i:i+self.batch_size], augment=False)
            descriptors = self.model.predict(batch_imgs)
            descriptors = descriptors.reshape(-1, descriptors.shape[-1])
            self.catalog_descriptors += list(descriptors)
        
        self.catalog_descriptors = np.array(self.catalog_descriptors)

    def get_query_scores(self, query_descriptors):
        matches = self.matcher.knnQueryBatch(query_descriptors, k=self.k_nn_deep)
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
        query_img = query_img[:, :, ::-1] / 255.0
        query_descriptors = self.model.predict(np.array([query_img]))
        query_descriptors = query_descriptors.reshape(-1, query_descriptors.shape[-1])
        scores = self.get_query_scores(query_descriptors)
        return scores
