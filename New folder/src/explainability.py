import shap
import torch

def shap_explain(model, background_data, test_data):
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(test_data)
    return shap_values
