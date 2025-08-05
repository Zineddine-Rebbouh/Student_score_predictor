
import numpy as np
from sklearn.model_selection import train_test_split

class FeatureEngineer:
    def __init__(self, data):
        self.data = data
    
    def prepare_features(self, target_column='Exam_Score', feature_columns=None):
        """Prepare features for model training"""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return False
        
        print("\n" + "="*50)
        print("🎯 FEATURE PREPARATION")
        print("="*50)
        
        # Select features
        if feature_columns is None:
            # Automatically select numerical columns except target
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numerical_cols if col != target_column]
        
        print(f"🔍 Selected features: {feature_columns}")
        print(f"🎯 Target variable: {target_column}")
        
        # Prepare X and y
        X = self.data[feature_columns]
        y = self.data[target_column]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\n📊 Data Split:")
        print(f"   • Training set: {self.X_train.shape[0]} samples")
        print(f"   • Test set: {self.X_test.shape[0]} samples")
        print(f"   • Features: {self.X_train.shape[1]}")
        
        return True
    