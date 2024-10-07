# main.py or notebook

import DataProcessor as dp
import ModelTrainer as mt
import Visualizer as vz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# Data Processing
DataProcessor = dp.DataProcessor()
DataProcessor.load_data()
DataProcessor.clean_data()
DataProcessor.feature_engineering()
df = DataProcessor.processed_data

# Visualization
Visualizer = vz.Visualizer()
Visualizer.plot_histograms(df, ['x', 'y', 'z'])
Visualizer.plot_kde(df, ['x', 'y', 'z'])

# Prepare data
X = df[['x', 'y', 'z', 'pctid', 'vibration_magnitude']]
y = df['wconfid']

# Split data
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test)

# Define models
models = [
    {
        'model': RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
        'params': {'n_estimators': 100, 'max_depth': None},
        'name': 'RandomForest_Model',
        'X_train': X_train_full,
        'X_test': X_test
    },
    {
        'model': SVC(kernel='rbf', C=1, gamma='scale', random_state=42),
        'params': {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
        'name': 'SVM_Model',
        'X_train': X_train_scaled,
        'X_test': X_test_scaled
    },
    {
        'model': xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1, subsample=1.0,
            use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'params': {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'subsample': 1.0},
        'name': 'XGBoost_Model',
        'X_train': X_train_full,
        'X_test': X_test
    }
]

# Train and evaluate models
for model_info in models:
    trainer = mt.ModelTrainer(
        model=model_info['model'],
        params=model_info['params'],
        model_name=model_info['name'],
        mlflow_experiment="Vibration_Configuration_Classification_refactorizado"
    )
    trainer.run_training(
        model_info['X_train'],
        model_info['X_test'],
        y_train_full,
        y_test
    )
