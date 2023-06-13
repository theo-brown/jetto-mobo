import pathlib
from typing import Any, Callable, Iterable, List, Mapping, Optional, Tuple, Union

import netCDF4
import pydantic
import torch
import yaml
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from jetto_tools import template as jetto_template


# Redefine the Base Model to allow arbitrary types,
# rather than just Python built-ins
class BaseModel(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True


# Config for surrogate model (Gaussian Process)
class SurrogateConfig(BaseModel):
    device: torch.device
    dtype: torch.dtype = torch.float64
    model: str
    kwargs: dict = {}

    @pydantic.validator("device", pre=True)
    def check_device(cls, device):
        # TODO: add NPU support
        # TODO: add support for "gpu:0" syntax
        supported_devices = ["cpu", "cuda"]
        if device not in supported_devices:
            raise ValueError(
                f"Device must be one of {supported_devices} (got {device})."
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device specified but CUDA is not available.")
        return torch.device(device)

    @pydantic.validator("dtype", pre=True)
    def check_dtype(cls, dtype):
        supported_types = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
        }
        if dtype not in supported_types.keys():
            raise ValueError(
                f"Data type must be one of {list(supported_types.keys())} (got {dtype})."
            )
        return supported_types[dtype]

    @pydantic.validator("model")
    def check_model(cls, model):
        # TODO: Can we set this to be the actual model class from Botorch?
        supported_models = ["ModelListGP", "MultiTaskGP", "SingleTaskGP"]
        if model not in supported_models:
            raise ValueError(f"Model must be one of {supported_models} (got {model})")
        return model


# Config for acquisition functions
class AcquisitionConfig(BaseModel):
    function: Callable
    raw_samples: int = 512
    n_restarts: int = 10
    mode: str = "joint"
    batch_limit: int = 5
    batch_size: int = 1
    max_iterations: int = 200
    n_sobol_samples: int = 256
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64
    kwargs: dict = {}

    @pydantic.validator("device", pre=True)
    def check_device(cls, device):
        # TODO: add NPU support
        # TODO: add support for "gpu:0" syntax
        supported_devices = ["cpu", "cuda"]
        if device not in supported_devices:
            raise ValueError(
                f"Device must be one of {supported_devices} (got {device})."
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device specified but CUDA is not available.")
        return torch.device(device)

    @pydantic.validator("dtype", pre=True)
    def check_dtype(cls, dtype):
        supported_types = {
            "float64": torch.float64,
            "float32": torch.float32,
            "float16": torch.float16,
        }
        if dtype not in supported_types.keys():
            raise ValueError(
                f"Data type must be one of {list(supported_types.keys())} (got {dtype})."
            )
        return supported_types[dtype]

    @pydantic.validator("function", pre=True)
    def check_function(cls, function_name):
        supported_functions = {
            "qNEI": qNoisyExpectedImprovement,
            "qNEHVI": qNoisyExpectedHypervolumeImprovement,
        }
        if function_name not in supported_functions.keys():
            raise ValueError(
                f"Acquisition function must be one of {list(supported_functions.keys())} (got {function_name})."
            )

        return supported_functions[function_name]

    @pydantic.validator("mode")
    def check_mode(cls, mode):
        supported_modes = ["joint", "sequential"]
        if mode not in supported_modes:
            raise ValueError(f"Mode must be one of {supported_modes} (got {mode}).")
        return mode

    @pydantic.validator(
        "raw_samples",
        "n_restarts",
        "batch_size",
        "batch_limit",
        "max_iterations",
        "n_sobol_samples",
    )
    def check_positivity(cls, v):
        if not v > 0:
            raise ValueError(
                f"Acquisition function parameters must be positive integers (got {v})"
            )
        return v


# Config for input functions
INPUT_FUNCTIONS = {}


def bo_input(f: Callable, name: Optional[str] = None):
    if name is None:
        name = f.__name__
    INPUT_FUNCTIONS[name] = f
    return f


class InputConfig(BaseModel):
    fixed_parameters: Optional[Mapping[str, float]] = None
    parameter_bounds: Mapping[str, Tuple[float, float]]
    function: Callable

    @pydantic.validator("parameter_bounds")
    def check_parameters(cls, parameter_bounds):
        for parameter_name, bounds in parameter_bounds.items():
            if not bounds[1] > bounds[0]:
                raise ValueError(
                    f"Upper bound must be greater than lower bound (got lower_bound={bounds[0]}, upper_bound={bounds[1]} for parameter {parameter_name})."
                )
        return parameter_bounds

    @pydantic.validator("function", pre=True)
    def parse_function(cls, function_name):
        # TODO: check that the function has the correct signature
        if function_name in INPUT_FUNCTIONS.keys():
            return INPUT_FUNCTIONS[function_name]
        else:
            raise ValueError(
                f"No registered function found with name {function_name}; did you forget to decorate it with @bo_input?"
            )


# Config for objective function (output)
OBJECTIVE_FUNCTIONS = {}


def bo_objective(f: Callable, name: Optional[str] = None):
    if name is None:
        name = f.__name__
    OBJECTIVE_FUNCTIONS[name] = f
    return f


class ObjectiveComponentConfig(BaseModel):
    name: str
    function: Callable[[netCDF4.Dataset, netCDF4.Dataset, Optional[Any]], float]
    weight: float
    value_on_failure: float
    kwargs: Mapping[str, Any]

    @pydantic.validator("function", pre=True)
    def parse_function(cls, function_name):
        # TODO: check that the function has the correct signature
        if function_name in OBJECTIVE_FUNCTIONS.keys():
            return OBJECTIVE_FUNCTIONS[function_name]
        else:
            raise ValueError(
                f"No function found with name {function_name}; did you forget to decorate it with @bo_objective?"
            )


class ObjectiveConfig(BaseModel):
    components: List[ObjectiveComponentConfig]
    as_scalar: bool = False


# Config for JETTO simulations
class SimulationConfig(BaseModel):
    template: jetto_template.Template
    image: pathlib.Path
    timelimit: int = -1

    @pydantic.validator("template", pre=True)
    def load_template(cls, path_to_template):
        return jetto_template.from_directory(path_to_template)

    @pydantic.validator("image")
    def check_image(cls, image):
        if not image.exists():
            raise ValueError(f"Image {image} does not exist.")
        if not image.suffix == ".sif":
            # TODO: More robust checking that it is indeed a JETTO .sif image
            raise ValueError(f"Image {image} is not a .sif Singularity image file.")
        return image

    @pydantic.validator("timelimit")
    def check_timelimit(cls, timelimit):
        if not timelimit > 0 and timelimit != -1:
            raise ValueError(
                f"Timelimit must either be -1 or greater than 0 (got {timelimit})"
            )
        return timelimit


# Config for the entire JETTO-MOBO run
class Config(BaseModel):
    config_file: Optional[pathlib.Path] = None
    output_directory: pathlib.Path
    n_optimisation_steps: int
    simulation: SimulationConfig
    input_: InputConfig
    objective: ObjectiveConfig
    surrogate: SurrogateConfig
    acquisition: AcquisitionConfig

    @pydantic.validator("config_file")
    def check_config_file(cls, config_file):
        if config_file is not None:
            if not config_file.exists():
                raise ValueError(f"Config .YAML file {config_file} not found")
        return config_file

    @pydantic.validator("output_directory")
    def create_output_dir(cls, output_directory):
        if not output_directory.exists():
            output_directory.mkdir(parents=True)
        return output_directory


def load_config(path: pathlib.Path) -> Config:
    config = _load_config_to_dict(path)
    config["config_file"] = path
    return Config(**config)


def _load_config_to_dict(path: pathlib.Path) -> dict:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
