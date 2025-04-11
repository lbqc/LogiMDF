import json
import os
import os.path
import pickle
import joblib
import pandas as pd
import torch
import yaml
import inspect

from .logger import GENERAL_SHELL_LOGGER


__func_name_needed_params__ = {}


def register_func_name_needed_params(func, name=None):
    existing_params = __func_name_needed_params__.get(name, __func_name_needed_params__.get(func))
    if existing_params:
        return existing_params
    
    signature = inspect.signature(func)
    param_names = list(signature.parameters.keys())
    
    if name:
        __func_name_needed_params__[name] = param_names
    else:
        __func_name_needed_params__[func] = param_names
    
    return param_names


def increment_path(path, folder_name, sep='', verbose=True, logger=GENERAL_SHELL_LOGGER):
    if not os.path.exists(os.path.join(path, folder_name)):
        if verbose and logger:
            logger.info(f"data will be saved to {path}/{folder_name}/")
        return folder_name
    usable_path = False
    for i in range(1, 9999):
        name = f'{path}/{folder_name}{sep}{i}'
        if not os.path.exists(name) or not os.path.isdir(name):
            folder_name = f'{folder_name}{sep}{i}'
            usable_path = True
            break
    if not usable_path:
        raise FileExistsError(f"folder_name: {folder_name}\tsep: {sep}\t range(1, 9999) is not usable")
    if verbose and logger:
        logger.info(f"data will be saved to {path}/{folder_name}/")
    return folder_name


def read_data_from_csv(filename, *args, **kwargs):
    if 'sep' not in kwargs:
        kwargs['sep'] = '\t'
        
    param_names = register_func_name_needed_params(pd.read_csv)
    with open(filename, 'r') as csvfile:
        data = pd.read_csv(csvfile, **{key: kwargs.pop(key) for key in param_names if key in kwargs})
    return data


def read_data_from_pickle(filename, *args, **kwargs):
    param_names = register_func_name_needed_params(pickle.load)
    with open(filename, 'rb') as f:
        data = pickle.load(f, **{key: kwargs.pop(key) for key in param_names if key in kwargs})
    return data


def read_data_from_yaml(filename, *args, **kwargs):
    with open(filename, 'r') as f:
        data = yaml.load(f, yaml.FullLoader)
    return data


def read_data_from_json(filename, *args, **kwargs):
    param_names = register_func_name_needed_params(json.load)
    with open(filename, 'r') as f:
        data = json.load(f, **{key: kwargs.pop(key) for key in param_names if key in kwargs})
    return data


def read_data_from_txt(filename, *args, **kwargs):
    with open(filename, 'r') as f:
        param_names = register_func_name_needed_params(f.readlines, 'readlines')
        data = f.readlines(**{key: kwargs.pop(key) for key in param_names if key in kwargs})
    data = [row.strip() for row in data]
    return data


def load_sklearn_model(filename, *args, **kwargs):
    param_names = register_func_name_needed_params(joblib.load)
    model = joblib.load(filename, **{key: kwargs.pop(key) for key in param_names if key in kwargs})
    return model


def load_pytorch_model(filename, *args, **kwargs):
    model = kwargs.pop('model')
    param_names = register_func_name_needed_params(torch.load)
    model_data = torch.load(filename, **{key: kwargs.pop(key) for key in param_names if key in kwargs})
    model.load_state_dict(model_data)
    return model


def save_data_to_csv(data, filename, *args, **kwargs):
    if not isinstance(data, pd.DataFrame):
        param_names = register_func_name_needed_params(pd.DataFrame)
        data = pd.DataFrame(data, **{key: kwargs.pop(key) for key in param_names if key in kwargs})
    param_names = register_func_name_needed_params(data.to_csv, name='to_csv')
    data.to_csv(filename, **{key: kwargs.pop(key) for key in param_names if key in kwargs})
    return data


def save_data_to_pickle(data, filename, *args, **kwargs):
    param_names = register_func_name_needed_params(pickle.dump)
    with open(filename, 'wb') as f:
        pickle.dump(data, f, **{key: kwargs.pop(key) for key in param_names if key in kwargs})
    return data


def save_data_to_json(data, filename, *args, **kwargs):
    param_names = register_func_name_needed_params(json.dump)
    with open(filename, 'w') as f:
        json.dump(data, f, **{key: kwargs.pop(key) for key in param_names if key in kwargs})
    return data


def save_data_to_yaml(data, filename, *args, **kwargs):
    param_names = register_func_name_needed_params(yaml.dump)
    with open(filename, "w") as f:
        yaml.dump(data, f, **{key: kwargs.pop(key) for key in param_names if key in kwargs})
    return data


def save_sklearn_model(model, filename, *args, **kwargs):
    param_names = register_func_name_needed_params(joblib.dump)
    joblib.dump(model, filename, **{key: kwargs.pop(key) for key in param_names if key in kwargs})


def save_pytorch_model(model, filename, *args, **kwargs):
    torch.save(model.state_dict(), filename)


__load_model_type2method__ = {
    'sklearn': load_sklearn_model,
    'pytorch': load_pytorch_model,
    'pkl': load_sklearn_model,
    'pt': load_pytorch_model
}


def load_model(filename, verbose=True, logger=GENERAL_SHELL_LOGGER, *args, **kwargs):
    """
    Load a machine learning model from a file.

    Args:
        filename (str): The path to the file containing the model.
        logger (Logger, optional): Logger object for logging. Defaults to GENERAL_SHELL_LOGGER.
        *args: Additional positional arguments to pass to the loading method.
        **kwargs: Additional keyword arguments to pass to the loading method, including 'model_type'.

    Returns:
        model: The loaded machine learning model.
    """
    model_type = kwargs.pop("model_type", None)
    if model_type is None:
        model_type = filename.split('.')[-1].lower()
    load_model_method = __load_model_type2method__.get(model_type, load_sklearn_model)
    model = load_model_method(filename, *args, **kwargs)

    if verbose and logger:
        logger.info(f'Loaded model from {filename}.')
    return model


__load_data_suffix2method__ = {
    'yaml': read_data_from_yaml,
    'csv': read_data_from_csv,
    'tsv': read_data_from_csv,
    'txt': read_data_from_txt,
    'json': read_data_from_json,
    'pkl': read_data_from_pickle
}


def load_data(filename, verbose=True, logger=GENERAL_SHELL_LOGGER, *args, **kwargs):
    """
    Load data from a file based on its suffix.

    Args:
        filename (str): The path to the file containing the data.
        logger (Logger, optional): Logger object for logging. Defaults to GENERAL_SHELL_LOGGER.
        *args: Additional positional arguments to pass to the loading method.
        **kwargs: Additional keyword arguments to pass to the loading method.

    Returns:
        data: The loaded data.
    """

    suffix = filename.split('.')[-1].lower()
    
    load_data_method = __load_data_suffix2method__.get(suffix, read_data_from_txt)
        
    data = load_data_method(filename, *args, **kwargs)

    if verbose and logger:
        logger.info(f'Loaded data from {filename}')
    return data


__save_model_type2method__ = {
    'sklearn': save_sklearn_model,
    'pytorch': save_pytorch_model,
    'pkl': save_sklearn_model,
    'pt': save_pytorch_model
}


def save_model(model, save_path, model_name, model_type=None, verbose=True, logger=GENERAL_SHELL_LOGGER, *args, **kwargs):
    """
    Save a machine learning model to a file.

    Args:
        model: The machine learning model to save.
        save_path (str): The directory where the model will be saved.
        model_name (str): The name of the model.
        model_type (str, optional): The type of the model (e.g., 'sklearn', 'pytorch'). If None, it will be extracted from 'model_name'.
        logger (Logger, optional): Logger object for logging. Defaults to GENERAL_SHELL_LOGGER.
        *args: Additional positional arguments to pass to the saving method.
        **kwargs: Additional keyword arguments to pass to the saving method.

    Returns:
        None
    """
    if model_type is None:
        model_name, model_type = model_name.split('.')
        model_type = model_type.lower()
        model_name = model_name.lower()
    
    os.makedirs(save_path, exist_ok=True)
    filename = os.path.join(save_path, f'{model_name}.{model_type}')
    
    save_model_method = __save_model_type2method__.get(model_type, save_sklearn_model)
    save_model_method(model, filename, *args, **kwargs)

    if verbose and logger:
        logger.info(f'Saved model to {filename}')
        

__save_data_suffix2method__ = {
    'yaml': save_data_to_yaml,
    'csv': save_data_to_csv,
    'tsv': save_data_to_csv,
    'json': save_data_to_json,
    'pkl': save_data_to_pickle,
}


def save_data(data, filename, verbose=True, logger=GENERAL_SHELL_LOGGER, *args, **kwargs):
    """
    Save data to a file based on its suffix.

    Args:
        data: The data to save.
        filename (str): The path to the file where the data will be saved.
        logger (Logger, optional): Logger object for logging. Defaults to GENERAL_SHELL_LOGGER.
        *args: Additional positional arguments to pass to the saving method.
        **kwargs: Additional keyword arguments to pass to the saving method.

    Returns:
        result: The result returned by the saving method.
    """
    suffix = filename.split('.')[-1].lower()
    file_dir = os.path.dirname(filename)
    os.makedirs(file_dir, exist_ok=True)
    
    save_data_method = __save_data_suffix2method__.get(suffix, save_data_to_json)
    result = save_data_method(data, filename, *args, **kwargs)
    
    if verbose and logger:
        logger.info(f'Saved {len(data)} rows to {filename}')

    return result


if __name__ == '__main__':
    print('hello world')
