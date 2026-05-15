"""
Custom Inference Script for SageMaker Endpoint

This script defines how the model is loaded and how predictions are made
when deployed to a SageMaker endpoint. Use this when you need custom
pre/post-processing at inference time.

SageMaker will call these functions:
    - model_fn: Load the model
    - input_fn: Deserialize input data
    - predict_fn: Run prediction
    - output_fn: Serialize output
"""

import json
import os

import numpy as np
import xgboost as xgb


def model_fn(model_dir):
    """
    Load the trained model from the model directory.

    Args:
        model_dir: Path to the directory containing model artifacts

    Returns:
        Loaded XGBoost model
    """
    model_path = os.path.join(model_dir, 'xgboost-model')
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize the incoming request body.

    Args:
        request_body: The body of the incoming request
        request_content_type: The content type of the request

    Returns:
        XGBoost DMatrix ready for prediction
    """
    if request_content_type == 'text/csv':
        # Parse CSV input
        lines = request_body.strip().split('\n')
        data = []
        for line in lines:
            values = [float(x) for x in line.split(',')]
            data.append(values)
        return xgb.DMatrix(np.array(data))

    elif request_content_type == 'application/json':
        # Parse JSON input
        payload = json.loads(request_body)
        if isinstance(payload, dict) and 'instances' in payload:
            data = payload['instances']
        elif isinstance(payload, list):
            data = payload
        else:
            raise ValueError(f"Unsupported JSON format: {type(payload)}")
        return xgb.DMatrix(np.array(data))

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Run prediction on the input data.

    Args:
        input_data: DMatrix from input_fn
        model: Model from model_fn

    Returns:
        Prediction probabilities
    """
    predictions = model.predict(input_data)
    return predictions


def output_fn(prediction, response_content_type):
    """
    Serialize the prediction output.

    Args:
        prediction: Output from predict_fn
        response_content_type: Desired response content type

    Returns:
        Serialized prediction
    """
    if response_content_type == 'text/csv':
        return '\n'.join([str(p) for p in prediction])

    elif response_content_type == 'application/json':
        return json.dumps({
            'predictions': [
                {
                    'probability': float(p),
                    'predicted_label': int(p > 0.5)
                }
                for p in prediction
            ]
        })

    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")
