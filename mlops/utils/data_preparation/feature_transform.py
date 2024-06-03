from sklearn.feature_extraction import DictVectorizer

def fit_features_transform(df, features_names):
    dicts = df[features_names].to_dict(orient='records')
    dv = DictVectorizer()
    dv.fit(dicts)

    return dv


def get_features_target(df, features_names, target_name, features_transform):
    dicts = df[features_names].to_dict(orient='records')
    X = features_transform.transform(dicts)

    y = df[target_name]

    return X, y