# src/utils/mlflow_logger.py

import mlflow

class MLflowLogger:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, run_name):
        self.run = mlflow.start_run(run_name=run_name)
        return self.run

    def log_params(self, params):
        mlflow.log_params(params)

    def log_metrics(self, metrics):
        mlflow.log_metrics(metrics)

    def log_artifact(self, artifact_path):
        mlflow.log_artifact(artifact_path)

    def log_model(self, model, model_name):
        mlflow.sklearn.log_model(model, model_name)

    def end_run(self):
        mlflow.end_run()
