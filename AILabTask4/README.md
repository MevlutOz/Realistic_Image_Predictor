# Realistic Image Predictor

A deep learning model that predicts realism scores for images using a fine-tuned ResNet50 architecture. The model analyzes image characteristics and assigns a numerical score between 0 and 1, where higher values indicate more realistic images.

## 🎯 Project Overview

This project implements a regression model that predicts how realistic an image appears. The model processes images and outputs a continuous value between 0 and 1, representing the realism score.

### Key Features

- Data preprocessing pipeline with augmentation
- Fine-tuned ResNet50 architecture for regression
- Test-time augmentation for improved prediction accuracy
- Comprehensive model evaluation metrics
- Batch processing capability for multiple images

## 📋 Requirements

```
tensorflow
numpy
matplotlib
scikit-learn
pillow
```

## 🗂️ Project Structure

```
.
├── data_preprocessing.py    # Data loading and preprocessing pipeline
├── model_design.py         # Model architecture definition
├── model_training.py       # Training pipeline with fine-tuning
├── model_evaluation.py     # Evaluation metrics and visualization
├── inference_pipeline.py   # Inference pipeline for predictions
└── main.py                # Main script for running predictions
```

## 🚀 Getting Started

### 1. Dataset Preparation

Images should follow the naming convention:
```
r<realistic_score>_<image_name>.png
```
Example: `r0.50_00228_05516.png` (realism score = 0.5)

### 2. Training the Model

```python
from src.model_training import train_model

# Train the model
model, history, history_fine, test_data = train_model()
```

### 3. Making Predictions

```python
from src.main import ImagePredictor

# Initialize predictor
predictor = ImagePredictor('src/final_model.h5')

# Make prediction for single image
score = predictor.predict('path/to/image.jpg')
print(f"Realism Score: {score:.3f}")
```

## 📊 Model Architecture

The model uses a ResNet50 backbone with custom regression head:
- Pre-trained ResNet50 base (weights from ImageNet)
- Global Average Pooling
- Dense layers (1024 -> 512 -> 1)
- L2 regularization and batch normalization
- Linear activation for final regression output

## 🔄 Training Pipeline

1. **Data Preprocessing**
   - Image resizing to 224x224
   - Data augmentation (rotation, flipping)
   - Train/validation/test split

2. **Training Process**
   - Initial training with frozen base layers
   - Fine-tuning of last 20 layers
   - Early stopping and model checkpointing
   - Learning rate scheduling

3. **Evaluation Metrics**
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R-squared score
   - Residual analysis

## 📈 Performance Visualization

The evaluation pipeline generates:
- Predictions vs. Actual Values plot
- Residual plots
- Error distribution histograms
- Training history curves

## 🛠️ Advanced Usage

### Test Time Augmentation

```python
from src.model_training import test_time_augmentation

# Get enhanced predictions using TTA
augmented_predictions = test_time_augmentation(model, X_test, num_augmentations=5)
```

### Batch Processing

```python
from src.inference_pipeline import InferencePipeline

pipeline = InferencePipeline('src/final_model.h5')
predictions = pipeline.batch_predict(['image1.jpg', 'image2.jpg', 'image3.jpg'])
```

