import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DataExplorer:
    def __init__(self, data):
        self.data = data
        
    def display_info(self):
        """Comprehensive data exploration"""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*50)
        print("🔍 DATA EXPLORATION")
        print("="*50)
        
        # Basic info
        print("\n📊 Dataset Info:")
        print(self.data.info())
        
        print("\n📈 Statistical Summary:")
        print(self.data.describe())
        
        print("\n🔍 Missing Values:")
        missing = self.data.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values found!")
        
        print("\n📋 Data Types:")
        print(self.data.dtypes)
        
        # Display first few rows
        print("\n👀 First 5 rows:")
        print(self.data.head())
        
    
    def visualize_data(self, target_column='Exam_Score'):
        """Create comprehensive visualizations"""
        if self.data is None:
            print("❌ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*50)
        print("📊 DATA VISUALIZATION")
        print("="*50)
        
        # Check if target column exists
        if target_column not in self.data.columns:
            print(f"❌ Target column '{target_column}' not found.")
            print(f"Available columns: {list(self.data.columns)}")
            return
        
        # Set up the plotting area
        plt.figure(figsize=(20, 15))
        
        # 1. Target variable distribution
        plt.subplot(3, 4, 1)
        plt.hist(self.data[target_column], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'Distribution of {target_column}')
        plt.xlabel(target_column)
        plt.ylabel('Frequency')
        
        # 2. Box plot of target variable
        plt.subplot(3, 4, 2)
        plt.boxplot(self.data[target_column])
        plt.title(f'Box Plot of {target_column}')
        plt.ylabel(target_column)
        
        # 3. Correlation heatmap
        plt.subplot(3, 4, 3)
        numerical_data = self.data.select_dtypes(include=[np.number])
        correlation_matrix = numerical_data.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix')
        
        # 4. Feature relationships with target
        numerical_cols = [col for col in numerical_data.columns if col != target_column]
        
        plot_num = 4
        for i, col in enumerate(numerical_cols[:8]):  # Show up to 8 features
            plt.subplot(3, 4, plot_num)
            plt.scatter(self.data[col], self.data[target_column], alpha=0.6)
            plt.xlabel(col)
            plt.ylabel(target_column)
            plt.title(f'{col} vs {target_column}')
            
            # Add correlation coefficient
            corr = self.data[col].corr(self.data[target_column])
            plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plot_num += 1
            if plot_num > 12:
                break
        
        plt.tight_layout()
        plt.show()
        
        # Feature importance based on correlation
        correlations = abs(correlation_matrix[target_column]).sort_values(ascending=False)
        print(f"\n🎯 Feature Correlations with {target_column}:")
        for feature, corr in correlations.items():
            if feature != target_column:
                print(f"   • {feature}: {corr:.3f}")