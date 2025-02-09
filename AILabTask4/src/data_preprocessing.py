import os
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self, img_size=(224, 224)):
        self.img_size = img_size
        self.datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True
        )

    def load_and_preprocess_image(self, image_path, augment=False):
        img = load_img(image_path, target_size=self.img_size)
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)

        if augment:
            img_array = self.datagen.random_transform(img_array)

        return img_array

    def extract_score_from_filename(self, filename):
        pattern = r'^r(\d+\.\d+)_'
        match = re.match(pattern, filename)
        if not match:
            raise ValueError(f"Filename {filename} doesn't match expected format")
        score = float(match.group(1))
        if not 0 <= score <= 1:
            raise ValueError(f"Score {score} not between 0 and 1")
        return score

    def get_image_base_name(self, filename):
        return "_".join(filename.split("_")[1:])

    def prepare_dataset(self, data_dir):
        grouped_images = defaultdict(list)

        for filename in os.listdir(data_dir):
            if filename.endswith('.png'):
                image_path = os.path.join(data_dir, filename)
                base_name = self.get_image_base_name(filename)

                try:
                    img_array = self.load_and_preprocess_image(image_path)
                    score = self.extract_score_from_filename(filename)
                    grouped_images[base_name].append((img_array, score))
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")

        self.analyze_dataset(grouped_images)
        return grouped_images

    def analyze_dataset(self, grouped_images):
        scores = [score for items in grouped_images.values() for _, score in items]
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(scores, bins=20, edgecolor='black')
        plt.xlabel('Realism Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Realism Scores')

        plt.subplot(1, 2, 2)
        plt.boxplot(scores)
        plt.ylabel('Realism Score')
        plt.title('Box Plot of Realism Scores')

        plt.show()

        print(f"Mean Score: {np.mean(scores):.3f}, Std Dev: {np.std(scores):.3f}")

    def split_dataset(self, grouped_images, test_size=0.2, val_size=0.2):
        base_names = list(grouped_images.keys())
        train_names, temp_names = train_test_split(base_names, test_size=(test_size + val_size), random_state=42)
        val_names, test_names = train_test_split(temp_names, test_size=val_size / (test_size + val_size), random_state=42)

        def collect_data(names, augment=False):
            X, y = [], []
            for name in names:
                for img, score in grouped_images[name]:  # img is already a numpy array
                    if augment:
                        img = self.datagen.random_transform(img)  # Apply augmentation here
                    X.append(img)
                    y.append(score)
            return np.array(X), np.array(y)

        X_train, y_train = collect_data(train_names, augment=True)
        X_val, y_val = collect_data(val_names)
        X_test, y_test = collect_data(test_names)

        return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    data_dir = "../dataset/images"
    processor = DataPreprocessor()

    print("Loading and preprocessing dataset...")
    grouped_images = processor.prepare_dataset(data_dir)

    print("Splitting dataset into train, validation, and test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_dataset(grouped_images)

    print(f"Dataset split:\nTrain: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
