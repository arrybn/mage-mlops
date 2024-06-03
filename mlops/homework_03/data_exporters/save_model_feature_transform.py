if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

import mlflow
import pickle
import os

def dump_pickle(obj, filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)

@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment('Mage Taxi')

    feature_transform, regressor = data

    with mlflow.start_run():
        mlflow.sklearn.log_model(regressor, artifact_path="models_mlflow")
        dump_pickle(feature_transform, "feature_transform.pkl")
        mlflow.log_artifact('feature_transform.pkl', 'feature_transform')