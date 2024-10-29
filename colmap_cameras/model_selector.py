"""
2024 Daniil Sinitsyn

Colmap camera models implemented in PyTorch
"""
import torch
from typing import List

from .models.colmap_models_list import colmap_models

def model_selector(model_name: str,
                   data: List[float],
                   dtype=torch.float32,
                   device='cpu'):
    models = {(model.model_name): model for model in colmap_models}
    if model_name not in models:
        supported_models = ', '.join(models.keys())
        raise ValueError(
            f"Model {model_name} is not supported.\n Supported models: {supported_models}"
        )

    Model = models[model_name]
    
    if Model.num_extra_params == -1:
        expected_data_length = Model.num_focal_params + Model.num_pp_params + 1
        if len(data) < expected_data_length:
            raise ValueError(
                f"Expected at least {expected_data_length} parameters, got {len(data)}")
        expected_data_length = len(data)
    else:
        expected_data_length = Model.num_focal_params + Model.num_pp_params + Model.num_extra_params + 2
    if len(data) != expected_data_length:
        raise ValueError(
            f"Expected {expected_data_length} parameters, got {len(data)}")
    
    if isinstance(data, torch.Tensor):
        image_shape = data[:2].to(dtype).to(device)
        data = data[2:].to(dtype).to(device)
    else:
        image_shape = torch.tensor(data[:2], dtype=dtype, device=device)
        data = torch.tensor(data[2:], dtype=dtype, device=device)
    return Model(data, image_shape)

def default_initialization(model_name: str, image_shape, device='cpu', ):
    models = {(model.model_name): model for model in colmap_models}
    if model_name not in models:
        supported_models = ', '.join(models.keys())
        raise ValueError(
            f"Model {model_name} is not supported.\n Supported models: {supported_models}"
        )
    if isinstance(image_shape, torch.Tensor):
        image_shape = image_shape.to(torch.float32).to(device)
    else:
        image_shape = torch.tensor(image_shape, dtype=torch.float32, device=device)
    Model = models[model_name]
    return Model.default_initialization(image_shape).to(device)
