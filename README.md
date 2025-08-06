Student Performance Prediction
About
This project predicts student exam scores using the Student Performance Factors dataset. It features data preprocessing, EDA, and modeling with linear regression, polynomial regression, and feature engineering. Artifacts (models, scalers, datasets, metrics) are saved for reproducibility. Built with Python, Pandas, Scikit-learn, and Joblib.
Project Structure

score_student_system.ipynb: Jupyter notebook with the complete workflow (data loading, cleaning, EDA, modeling, and artifact saving).
models/: Contains saved artifacts:
linear_regression/: Linear regression model, scaler, datasets, feature names, metrics, and visualizations.
polynomial_regression/: Best polynomial model, datasets, feature names, and metrics.
feature_engineered/: Feature-engineered model, scaler, datasets, feature names, metrics, and feature selector (if applicable).


project_documentation.md: Detailed project documentation.

Setup

Clone the Repository:
git clone https://github.com/your-username/student-performance-prediction.git
cd student-performance-prediction


Install Dependencies:Ensure Python 3.12.4 is installed. Create a virtual environment and install required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Download the Dataset:

Obtain StudentPerformanceFactors.csv from Kaggle.
Place it in the data/raw_data/ directory.


Install Jupyter (if not already installed):
pip install jupyter



Usage

Run the Notebook:Start Jupyter Notebook and open score_student_system.ipynb:
jupyter notebook

Execute the cells sequentially to perform data preprocessing, EDA, model training, and evaluation. Artifacts are saved in the models/ directory.

Load Saved Models:Use the saved models for predictions:
import joblib
import pandas as pd

# Example: Load linear regression model
model = joblib.load('models/linear_regression/lr_model.joblib')
scaler = joblib.load('models/linear_regression/scaler.joblib')
X_test = pd.read_csv('models/linear_regression/X_test.csv')
X_test_scaled = scaler.transform(X_test)
predictions = model.predict(X_test_scaled)


View Results:

Check performance metrics in models/*/performance_metrics.csv.
Visualize results using saved plots (e.g., models/linear_regression/actual_vs_predicted.png).



Requirements

Python 3.12.4
Libraries: pandas, matplotlib, seaborn, scikit-learn, statsmodels, joblib
See requirements.txt for exact versions:pip install pandas matplotlib seaborn scikit-learn statsmodels joblib



Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a new branch (git checkout -b feature/your-feature).
Make changes and commit (git commit -m "Add your feature").
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

Please ensure code follows the existing style and includes tests where applicable.
License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Dataset: Student Performance Factors
Libraries: Pandas, Scikit-learn, Matplotlib, Seaborn, Statsmodels, Joblib
