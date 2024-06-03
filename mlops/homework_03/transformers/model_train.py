if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

from mlops.utils.data_preparation.feature_transform import fit_features_transform, get_features_target

from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import pandas as pd

@transformer
def transform(data: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], *args, **kwargs):
    """
    Template code for a transformer block.

    Add more parameters to this function if this block has multiple parent blocks.
    There should be one parameter for each output variable from each parent block.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    df = data[0]

    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']
    target = kwargs.get('target')

    feratures_transform = fit_features_transform(df, categorical + numerical)

    X, y = get_features_target(df, categorical + numerical, target, feratures_transform)

    regressor = LinearRegression()
    regressor.fit(X, y)

    print(f'RMSE: {root_mean_squared_error(y, regressor.predict(X))}')

    print(f'Model intercept: {regressor.intercept_}')

    return feratures_transform, regressor 


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'