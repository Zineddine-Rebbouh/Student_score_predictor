from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

class ModelTrainer:
    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer
        
    def train_linear_model(self):
        """Train linear regression model"""
        if self.X_train is None:
            print("❌ Please prepare features first.")
            return
        
        print("\n" + "="*50)
        print("🤖 LINEAR REGRESSION TRAINING")
        print("="*50)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Train linear regression
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train_scaled, self.y_train)
        
        # Make predictions
        y_train_pred = self.linear_model.predict(X_train_scaled)
        y_test_pred = self.linear_model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        print("📊 Linear Regression Results:")
        print(f"   • Training R²: {train_r2:.4f}")
        print(f"   • Test R²: {test_r2:.4f}")
        print(f"   • Training RMSE: {train_rmse:.4f}")
        print(f"   • Test RMSE: {test_rmse:.4f}")
        print(f"   • Training MAE: {train_mae:.4f}")
        print(f"   • Test MAE: {test_mae:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'coefficient': self.linear_model.coef_,
            'abs_coefficient': abs(self.linear_model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        print("\n🎯 Feature Importance (Linear Model):")
        for _, row in feature_importance.iterrows():
            print(f"   • {row['feature']}: {row['coefficient']:.4f}")
        
        return y_test_pred
    
    def train_polynomial_model(self, degree=2):
        """Train polynomial regression model"""
        if self.X_train is None:
            print("❌ Please prepare features first.")
            return
        
        print(f"\n" + "="*50)
        print(f"🚀 POLYNOMIAL REGRESSION TRAINING (Degree {degree})")
        print("="*50)
        
        # Create polynomial pipeline
        self.poly_model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=degree)),
            ('regressor', LinearRegression())
        ])
        
        # Train polynomial regression
        self.poly_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_train_pred = self.poly_model.predict(self.X_train)
        y_test_pred = self.poly_model.predict(self.X_test)
        
        # Calculate metrics
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        print(f"📊 Polynomial Regression Results (Degree {degree}):")
        print(f"   • Training R²: {train_r2:.4f}")
        print(f"   • Test R²: {test_r2:.4f}")
        print(f"   • Training RMSE: {train_rmse:.4f}")
        print(f"   • Test RMSE: {test_rmse:.4f}")
        print(f"   • Training MAE: {train_mae:.4f}")
        print(f"   • Test MAE: {test_mae:.4f}")
        
        return y_test_pred
