from fastapi import FastAPI
import json
from functions import load_model, get_features, get_scaled_features, get_predictions, get_shap_explainer, \
    get_feature_description, get_feature_importance

app = FastAPI()

model = load_model()
X = get_features()
X_scaled = get_scaled_features(X)
y, y_proba = get_predictions(model, X_scaled)

data = X.copy()
data['TARGET_PROBA'] = y_proba
data['TARGET'] = y

explainer = get_shap_explainer(model, X_scaled)
feature_description = get_feature_description(X)


@app.get("/")
async def root():
    return {"message": "Connection established!"}


@app.get("/id")
async def get_all_id():
    """
    :return: List of all SK_ID_CURR (ID of loan in our sample) contained in our data.
    """
    return json.dumps(X.index.to_list())


@app.get("/data")
async def get_all_data():
    """
    :return: 795 features used for prediction and the predicted variable 'TARGET': {0: 'Good client', 1: 'Bad client'} \
    for all loans.
    """
    return data.to_json()


@app.get("/data/{SK_ID_CURR}")
async def get_data_from_id(SK_ID_CURR: int):
    """
    :param SK_ID_CURR: ID of loan in our sample.
    :return: 795 features used for prediction and the predicted variable 'TARGET': {0: 'Good client', 1: 'Bad client'} \
    for a specific loan.
    """
    return data.loc[SK_ID_CURR].to_json()


@app.get("/metadata/{SK_ID_CURR}")
async def get_metadata_from_id(SK_ID_CURR: int):
    """
    :param SK_ID_CURR: ID of loan in our sample.
    :return: For a specific individual, returns Shapley value and description for each of the 795 features used \
    for prediction.
    """
    return get_feature_importance(explainer, X_scaled, SK_ID_CURR, feature_description).to_json()


@app.get("/target/{SK_ID_CURR}")
async def get_target_from_id(SK_ID_CURR: int):
    """
    :param SK_ID_CURR: ID of loan in our sample.
    :return: variable 'TARGET': {0: 'Good client', 1: 'Bad client'} for a specific loan.
    """
    return data.loc[SK_ID_CURR][['TARGET']].to_json()


@app.get("/target_proba/{SK_ID_CURR}")
async def get_target_proba_from_id(SK_ID_CURR: int):
    """
    :param SK_ID_CURR: ID of loan in our sample.
    :return: variable 'TARGET_PROBA': probability of gain for a specific loan.
    """
    return data.loc[SK_ID_CURR][['TARGET_PROBA']].to_json()
