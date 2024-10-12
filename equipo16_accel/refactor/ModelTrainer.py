# src/models/model_trainer.py
import os

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    def __init__(self, model, params, model_name, mlflow_experiment):
        self.model = model
        self.params = params
        self.model_name = model_name
        self.mlflow_experiment = mlflow_experiment
        self.run = None

    def _train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def predict(self, x_test):
        return self.model.predict(x_test)

    def _evaluate(self, y_test, y_pred):
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        return metrics

    def _log_metrics(self, metrics):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    
    def _log_artifacts(self, y_test, y_pred):
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
    
    def _log_model(self):
        if hasattr(self.model, 'save_model'):
            mlflow.xgboost.log_model(self.model, self.model_name)
        else:
            mlflow.sklearn.log_model(self.model, self.model_name)
    
    def run_training(self, x_train, x_test, y_train, y_test):
        mlflow.set_experiment(self.mlflow_experiment)

        if os.environ.get('MLFLOW_TRACKING_URI') is not None:
            # Set MLFLOW server if found by environment variable
            mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI'))

        with mlflow.start_run(run_name=self.model_name) as run:
            self.run = run
            # Log parameters
            mlflow.log_params(self.params)
            
            # Train the model
            self._train(x_train, y_train)
            
            # Make predictions
            y_pred = self.predict(x_test)
            
            # Evaluate the model
            metrics = self._evaluate(y_test, y_pred)
            self._log_metrics(metrics)
            
            # Log artifacts
            self._log_artifacts(y_test, y_pred)
            
            # Log the model
            self._log_model()
            
            print(f"{self.model_name} Model Metrics: {metrics}")
            print(f"Run ID: {run.info.run_id}")
