"""
Author: Martin Zelenák (xzelen27@stud.fit.vutbr.cz)
Description: Configuration parsing utilities.
Date: 2025-05-14
"""


from dataclasses import dataclass
from typing import Tuple

from .predictionModel import ModelParams, MODEL_TYPES

@dataclass
class PredictionConfig:
    model_params: ModelParams
    load_model: bool
    save_model: bool
    save_after_n_learning_steps: int
    models_folder: str
    predict_on_new_state: bool
    periodic_prediction: Tuple[bool, int]

    @staticmethod
    def from_dict(config: dict) -> 'PredictionConfig':
        return PredictionConfig(
            model_params=ModelParams.from_dict(config['model_params']),
            load_model=bool(config['load_model']),
            save_model=bool(config['save_model']),
            save_after_n_learning_steps=int(config['save_after_n_learning_steps']),
            models_folder=str(config['models_folder']),
            predict_on_new_state=bool(config['predict_on_new_state']),
            periodic_prediction=(bool(config['periodic_prediction']['enabled']), int(config['periodic_prediction']['period']))
        )
    
default_prediction_config = PredictionConfig(
    model_params=ModelParams(MODEL_TYPES[0], 32, 2, 0.001, 1, True), 
    load_model=False, 
    save_model=False, 
    save_after_n_learning_steps=0, 
    models_folder='./saved_models', 
    predict_on_new_state=False, 
    periodic_prediction=(True, 5)
)
