import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import r2_score
import seaborn as sns
from scipy import stats
from src.data_preprocessing import DataPreprocessor
from src.model_training import test_time_augmentation


class ModelEvaluator:
    def __init__(self, model_path='final_model.h5'):
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        self.metrics = {}

    def compute_metrics(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        r2 = r2_score(y_true, y_pred)

        self.metrics = {
            'MSE': mse,
            'MAE': mae,
            'R-squared': r2,
            'RMSE': np.sqrt(mse)
        }
        return self.metrics

    def plot_residuals(self, y_true, y_pred, save_path=None):
        residuals = y_true - y_pred.flatten()  # Ensure correct shape

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted Values')

        # Residual Distribution
        sns.histplot(residuals, kde=True, ax=axes[0, 1])
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].set_xlabel('Residual Value')

        # Actual vs Predicted
        axes[1, 0].scatter(y_true, y_pred, alpha=0.5)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
        axes[1, 0].set_xlabel('Actual Values')
        axes[1, 0].set_ylabel('Predicted Values')
        axes[1, 0].set_title('Actual vs Predicted Values')

        # Q-Q plot for normality check
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot of Residuals')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def evaluate_with_tta(self, X_test, y_test, num_augmentations=5):
        """Evaluate model with Test Time Augmentation"""
        try:
            # Get predictions with TTA
            print("Performing Test Time Augmentation...")
            tta_predictions = test_time_augmentation(self.model, X_test, num_augmentations)

            # Get regular predictions
            print("Performing regular prediction...")
            regular_predictions = self.model.predict(X_test, verbose=0)

            # Compute metrics
            print("Computing metrics...")
            tta_metrics = self.compute_metrics(y_test, tta_predictions)
            regular_metrics = self.compute_metrics(y_test, regular_predictions)

            return {
                'TTA': tta_metrics,
                'Regular': regular_metrics,
                'Predictions': {
                    'TTA': tta_predictions,
                    'Regular': regular_predictions
                }
            }
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            raise


def main():
    try:
        # Initialize preprocessor and load test data
        print("Loading and preprocessing data...")
        preprocessor = DataPreprocessor()
        data_dir = '../dataset/images'
        grouped_images = preprocessor.prepare_dataset(data_dir)
        _, _, X_test, _, _, y_test = preprocessor.split_dataset(grouped_images)

        # Initialize evaluator
        print("Initializing model evaluator...")
        evaluator = ModelEvaluator()

        # Perform evaluation with and without TTA
        print("Performing model evaluation...")
        evaluation_results = evaluator.evaluate_with_tta(X_test, y_test)

        # Print results
        print("\nRegular Evaluation Metrics:")
        for metric, value in evaluation_results['Regular'].items():
            print(f"{metric}: {value:.4f}")

        print("\nTest Time Augmentation Metrics:")
        for metric, value in evaluation_results['TTA'].items():
            print(f"{metric}: {value:.4f}")

        # Generate residual plots
        print("\nGenerating residual plots...")
        evaluator.plot_residuals(
            y_test,
            evaluation_results['Predictions']['Regular'],
            save_path='regular_residuals.png'
        )
        evaluator.plot_residuals(
            y_test,
            evaluation_results['Predictions']['TTA'],
            save_path='tta_residuals.png'
        )

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()