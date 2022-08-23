import os
import yaml
import joblib
from src import clean_data, build_features

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def api_response(request_dict):
    params_path = "params.yaml"
    config = read_params(params_path)

    request_text = request_dict.text

    request_text_clean = clean_data.DataLoader(config_path=params_path)._preprocess_text(request_text)

    request_text_features = build_features.BuildFeatures(config_path=params_path)._build_features_text(request_text_clean)
    
    model_dir = os.path.join(config["model_dir"], "model.joblib")
    model = joblib.load(model_dir)
    prediction = model.predict(request_text_features).tolist()[0]
    prediction_proba = model.predict_proba(request_text_features)[:,1].tolist()[0]

    if prediction == 1:
        toxic = True
    elif prediction == 0:
        toxic = False

    return toxic, prediction_proba