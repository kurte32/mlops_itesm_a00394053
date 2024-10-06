# src/models/model_trainer.py

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelTrainer:
    def __init__(self, model, params, model_name, mlflow_experiment):
        self.model = model
        self.params = params
        self.model_name = model_name
        self.mlflow_experiment = mlflow_experiment
        self.run = None

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, y_test, y_pred):
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        return metrics
    
    def log_metrics(self, metrics):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    
    def log_artifacts(self, y_test, y_pred):
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=False)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close(fig)
    
    def log_model(self):
        if hasattr(self.model, 'save_model'):
            mlflow.xgboost.log_model(self.model, self.model_name)
        else:
            mlflow.sklearn.log_model(self.model, self.model_name)
    
    def run_training(self, X_train, X_test, y_train, y_test):
        mlflow.set_experiment(self.mlflow_experiment)
        with mlflow.start_run(run_name=self.model_name) as run:
            self.run = run
            # Log parameters
            mlflow.log_params(self.params)
            
            # Train the model
            self.train(X_train, y_train)
            
            # Make predictions
            y_pred = self.predict(X_test)
            
            # Evaluate the model
            metrics = self.evaluate(y_test, y_pred)
            self.log_metrics(metrics)
            
            # Log artifacts
            self.log_artifacts(y_test, y_pred)
            
            # Log the model
            self.log_model()
            
            print(f"{self.model_name} Model Metrics: {metrics}")
            print(f"Run ID: {run.info.run_id}")
