from sklearn.metrics import mean_absolute_error , r2_score, mean_squared_error
import numpy as np


class ModelEvaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self):
        """Evaluate the model using various metrics"""
        if self.model is None:
            print("❌ No model found. Please train a model first.")
            return

        print("\n" + "="*50)
        print("📊 MODEL EVALUATION")
        print("="*50)

        # Make predictions
        y_pred = self.model.predict(self.X_test)

        # Calculate metrics
        test_r2 = r2_score(self.y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        test_mae = mean_absolute_error(self.y_test, y_pred)

        print("📈 Evaluation Metrics:")
        print(f"   • Test R²: {test_r2:.4f}")
        print(f"   • Test RMSE: {test_rmse:.4f}")
        print(f"   • Test MAE: {test_mae:.4f}")

        return y_pred
    
    
    