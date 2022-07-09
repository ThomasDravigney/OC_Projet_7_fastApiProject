import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import shap


def load_model():
    return pickle.load(open('model.pickle', 'rb'))


def get_features():
    X = pd.read_parquet('test_data_sample.parquet')
    X.drop('TARGET', axis=1, inplace=True)
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    return X


def get_scaled_features(X):
    return pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)


def get_predictions(model, X_scaled):
    return model.predict(X_scaled), model.predict_proba(X_scaled)[:,0]


def get_shap_explainer(model, X_scaled):
    return shap.TreeExplainer(model, data=X_scaled)


def get_feature_description(X):
    desc_df = pd.read_csv('feature_description.csv', encoding_errors='replace')
    desc_df['Row New'] = desc_df['Row'].apply(lambda x: pd.DataFrame(X.columns)[X.columns.str.contains(x)][0].tolist())
    return desc_df


def match_feature_names(row, feature_description):
    feature_names = list()
    for i in feature_description.index:
        if row in feature_description['Row New'][i]:
            feature_names.append(feature_description['Description'][i])
    return feature_names


def get_feature_importance(explainer, X_scaled, SK_ID_CURR, feature_description):
    shap_values = explainer.shap_values(X_scaled.loc[SK_ID_CURR])
    feature_importance = pd.DataFrame({'feature_name': X_scaled.columns, 'feature_importance': shap_values})
    feature_importance['feature_description'] = feature_importance['feature_name'].apply(match_feature_names, feature_description=feature_description)
    feature_importance.sort_values(by='feature_importance', ascending=False, key=abs, inplace=True)
    return feature_importance
