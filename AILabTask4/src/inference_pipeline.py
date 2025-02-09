import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np


class InferencePipeline:
    def __init__(self, model_path, img_size=(224, 224)):
        # Load trained model
        self.model = tf.keras.models.load_model(model_path)
        self.img_size = img_size

    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Prevent NaNs
        img_array = np.nan_to_num(img_array)
        return img_array

    def predict(self, image_path):
        img_array = self.preprocess_image(image_path)
        prediction = self.model.predict(img_array, verbose=0)

        # Ensure predictions stay in [0,1] range
        return float(np.clip(prediction[0][0], 0, 1))

    def batch_predict(self, image_paths):
        img_arrays = np.vstack([self.preprocess_image(path) for path in image_paths])
        predictions = self.model.predict(img_arrays, verbose=0)

        # Ensure predictions stay in [0,1] range
        return [float(np.clip(pred, 0, 1)) for pred in predictions.flatten()]
