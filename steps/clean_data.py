import numpy as np

class DataCleaner:
    def __init__(self, data):
        self.data = data


    def clean_data(self):
        """Clean the dataset and handle missing values"""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*50)
        print("🧹 DATA CLEANING")
        print("="*50)
        
        initial_shape = self.data.shape
        
        # Handle missing values
        missing_before = self.data.isnull().sum().sum()
        
        # Fill numerical missing values with median
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.data[col].isnull().sum() > 0:
                median_val = self.data[col].median()
                self.data[col].fillna(median_val, inplace=True)
                print(f"✅ Filled {col} missing values with median: {median_val}")
        
        # Fill categorical missing values with mode
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].isnull().sum() > 0:
                mode_val = self.data[col].mode()[0]
                self.data[col].fillna(mode_val, inplace=True)
                print(f"✅ Filled {col} missing values with mode: {mode_val}")
        
        missing_after = self.data.isnull().sum().sum()
        
        # Remove duplicates
        duplicates_before = self.data.duplicated().sum()
        self.data.drop_duplicates(inplace=True)
        duplicates_after = self.data.duplicated().sum()
        
        print(f"\n📊 Cleaning Summary:")
        print(f"   • Initial shape: {initial_shape}")
        print(f"   • Final shape: {self.data.shape}")
        print(f"   • Missing values: {missing_before} → {missing_after}")
        print(f"   • Duplicates removed: {duplicates_before}")