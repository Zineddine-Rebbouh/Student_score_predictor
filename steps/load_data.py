import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Load and display basic information about the dataset"""
        try:
            self.data = pd.read_csv(self.file_path)
            print("✅ Dataset loaded successfully!")
            return self.data
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

