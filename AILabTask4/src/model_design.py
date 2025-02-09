# model_design.py
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf


class ModelBuilder:
    def __init__(self, optimizer='sgd', fine_tune=False):
        self.model = None
        self.optimizer = optimizer
        self.fine_tune = fine_tune  # Allow fine-tuning of deeper layers

    def r2_score_metric(self, y_true, y_pred):
        SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
        SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
        return 1 - (SS_res / (SS_tot + tf.keras.backend.epsilon()))

    def build_model(self, input_shape=(224, 224, 3)):
        # Load pretrained ResNet50 without top layers
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

        # Regression head
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        predictions = Dense(1, activation='linear')(x)

        # Create full model
        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze all base layers initially
        for layer in base_model.layers:
            layer.trainable = False

        # Optionally unfreeze top layers for fine-tuning
        if self.fine_tune:
            for layer in base_model.layers[-10:]:  # Unfreeze last 10 layers
                layer.trainable = True

        # Choose optimizer
        if self.optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        else:
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

        # Compile the model
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError(), self.r2_score_metric]
        )

        return self.model
