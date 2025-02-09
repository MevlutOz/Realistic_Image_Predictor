import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from src.data_preprocessing import DataPreprocessor


def build_model(input_shape=(224, 224, 3), fine_tune=False):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    # Fine-tune last 20 layers if required
    if fine_tune:
        for layer in base_model.layers[-20:]:
            layer.trainable = True
    else:
        for layer in base_model.layers:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    predictions = Dense(1, activation='linear')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )

    return model


def train_model():
    preprocessor = DataPreprocessor()
    data_dir = '../dataset/images'
    grouped_images = preprocessor.prepare_dataset(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(grouped_images)

    # Data Augmentation for training
    train_datagen = ImageDataGenerator(rotation_range=30, horizontal_flip=True)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

    model = build_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('check_model.h5', monitor='val_loss', save_best_only=True)

    history = model.fit(
        train_generator,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=100,
        callbacks=[early_stopping, checkpoint]
    )

    # Fine-tuning step: Unfreeze last 20 layers
    model = build_model(fine_tune=True)
    history_fine = model.fit(
        train_generator,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=50,
        callbacks=[early_stopping, checkpoint]
    )

    model.save('final_model.h5')

    return model, history, history_fine, (X_test, y_test)


def test_time_augmentation(model, X_test, num_augmentations=5):
    augmented_preds = []
    test_datagen = ImageDataGenerator(rotation_range=15, horizontal_flip=True)

    for _ in range(num_augmentations):
        iterator = test_datagen.flow(X_test, batch_size=len(X_test), shuffle=False)
        X_test_aug = next(iterator)
        preds = model.predict(X_test_aug, verbose=0)  # Added verbose=0 to reduce output
        augmented_preds.append(preds)

    final_preds = np.mean(augmented_preds, axis=0)
    return final_preds


if __name__ == "__main__":
    model, history, history_fine, test_data = train_model()
    print("\nTraining History:")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")
    print(f"Best validation MAE: {min(history.history['val_mean_absolute_error']):.4f}")

    X_test, y_test = test_data
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print("\nTest Set Performance:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

