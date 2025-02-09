# main.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import os
import model_evaluation as md

class ImagePredictor:
    def __init__(self, model_path):
        # Load the trained model
        self.model = tf.keras.models.load_model(model_path)

        # Recompile the model with the same loss and metrics
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()]
        )

        # Set image size to 224x224 (ResNet50 required input size)
        self.img_size = (224, 224)

    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction
        - Resize to 224x224 (ResNet50 required input size)
        - Convert to array
        - Preprocess for ResNet50
        """
        print(f"Processing image {image_path}")
        print(f"Resizing to {self.img_size}")

        # Load and resize image to 224x224
        img = load_img(image_path, target_size=self.img_size)

        # Convert to array and preprocess
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        return img_array

    def predict(self, image_path):
        try:
            # Preprocess image (includes resizing to 224x224)
            img_array = self.preprocess_image(image_path)

            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)
            return float(prediction[0][0])
        except Exception as e:
            print(f"Error predicting for {image_path}: {str(e)}")
            return None


def main():

    model_path = 'final_model.h5'

    # Initialize predictor
    try:
        predictor = ImagePredictor(model_path)
        print("Model loaded successfully!")
        print("Image size set to 224x224 (ResNet50 required input size)")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Directory containing images to predict
    image_dir = '../dataset/test'

    # Process all images in the directory
    if os.path.isdir(image_dir):
        print(f"\nProcessing images from directory: {image_dir}")
        print("All images will be resized to 224x224")

        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(image_dir, filename)
                score = predictor.predict(image_path)
                if score is not None:
                    print(f"Image: {filename} -> Realism Score: {score:.3f}")
    else:
        # If image_dir is a single image file
        if os.path.isfile(image_dir) and image_dir.lower().endswith(('.png', '.jpg', '.jpeg')):
            print("Processing single image")
            print("Image will be resized to 224x224")
            score = predictor.predict(image_dir)
            if score is not None:
                print(f"Image: {os.path.basename(image_dir)} -> Realism Score: {score:.3f}")
        else:
            print("Please provide a valid image path or directory")


if __name__ == "__main__":
    main()