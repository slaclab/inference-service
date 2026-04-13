import os
import logging
import sys
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request
from pydantic import BaseModel
import uvicorn
import mlflow
from mlflow.tracking import MlflowClient
from lume_model.models import TorchModel

# Import torch and numpy at the top for efficiency
try:
    import torch
except ImportError:
    torch = None

try:
    import numpy as np
except ImportError:
    np = None

# Set up logging
logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# Model configuration
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", None)
DEFAULT_MODEL_VERSION = os.environ.get("MODEL_VERSION", None)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)



# Request/Response models
class PredictionRequest(BaseModel):
    inputs: Dict[str, float]


class PredictionResponse(BaseModel):
    outputs: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    inputs_list: List[Dict[str, float]]


class BatchPredictionResponse(BaseModel):
    outputs_list: List[Dict[str, float]]
    batch_size: int


class ModelInputsResponse(BaseModel):
    input_names: List[str]
    input_variables: Dict[str, Any]


class ModelOutputsResponse(BaseModel):
    output_names: List[str]
    output_variables: Dict[str, Any]


class LoadModelRequest(BaseModel):
    model_name: str
    model_version: Optional[str] = None


class ModelInfo(BaseModel):
    loaded: bool
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    run_id: Optional[str] = None
    input_names: Optional[List[str]] = None
    output_names: Optional[List[str]] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

class VariableTypesResponse(BaseModel):
    input_types: Dict[str, str]
    output_types: Dict[str, str]




def download_model_artifacts(model_name: str, model_version: Optional[str] = None) -> tuple[str, str]:
    """
    Download model artifacts from MLflow
    
    Returns
    -------
    tuple[str, str]
        (run_id, artifact_path) - path to the downloaded artifacts directory
    """
    client = MlflowClient()
    
    # Get model version info
    if model_version is None or model_version.lower() == "latest":
        versions = client.get_latest_versions(model_name)
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        model_version_obj = versions[0]
    elif model_version.isdigit():
        model_version_obj = client.get_model_version(model_name, model_version)
    else:
        # Stage name
        versions = client.get_latest_versions(model_name, stages=[model_version])
        if not versions:
            raise ValueError(f"No version found in stage '{model_version}' for model '{model_name}'")
        model_version_obj = versions[0]
    
    run_id = model_version_obj.run_id
    
    logger.info(f"Downloading artifacts from run_id: {run_id}")
    
    # Download artifacts to a temporary directory for better cleanup control
    temp_dir = tempfile.mkdtemp(prefix="mlflow_artifacts_")
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id, dst_path=temp_dir)

    logger.info(f"Artifacts downloaded to: {artifact_path}")
    
    return run_id, artifact_path


def find_yaml_config(artifact_path: str) -> str:
    """
    Find the LUME model YAML config in the artifact directory
    
    Parameters
    ----------
    artifact_path : str
        Path to the downloaded artifacts
    
    Returns
    -------
    str
        Path to the model config YAML file
    """
    artifact_dir = Path(artifact_path)
    
    yaml_files = list(artifact_dir.glob("*.yaml")) + list(artifact_dir.glob("*.yml"))
    
    if yaml_files:
        logger.info(f"Found YAML file: {yaml_files[0]}")
        return str(yaml_files[0])
    
    # Search in subdirectories
    yaml_files = list(artifact_dir.rglob("*.yaml")) + list(artifact_dir.rglob("*.yml"))
    
    if yaml_files:
        logger.info(f"Found YAML file in subdirectory: {yaml_files[0]}")
        return str(yaml_files[0])
    
    raise FileNotFoundError(f"No YAML config file found in {artifact_path}")


def load_lume_model(model_name: str, model_version: Optional[str] = None) -> TorchModel:
    """
    Load a LUME TorchModel from MLflow artifacts
    
    Parameters
    ----------
    model_name : str
        Name of the registered model in MLflow
    model_version : str, optional
        Version number, stage name, or "latest"

    Returns the loaded model.
    """
    
    try:
        logger.info(f"Loading LUME model '{model_name}' version '{model_version}'...")
        
        # Download artifacts from MLflow
        run_id, artifact_path = download_model_artifacts(model_name, model_version)
        
        # Find the YAML config file
        yaml_config_path = find_yaml_config(artifact_path)
        
        logger.info(f"Loading TorchModel from: {yaml_config_path}")
        
        # Load the LUME TorchModel
        model = TorchModel(yaml_config_path)
        
        # Set input validation to warn once when model is loaded
        model.input_validation_config = {k: "warn" for k in model.input_names}
        
        # Store metadata on the model object
        model._run_id = run_id
        model._model_name = model_name
        model._model_version = model_version
        model._config_path = yaml_config_path
        
        logger.info(f" LUME model loaded successfully!")
        logger.info(f"  Model name: {model_name}")
        logger.info(f"  Model version: {model_version}")
        logger.info(f"  Run ID: {run_id}")
        logger.info(f"  Config path: {yaml_config_path}")
        logger.info(f"  Input variables: {model.input_names}")
        logger.info(f"  Output variables: {model.output_names}")
        
        return model
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}", exc_info=True)
        raise


def clean_output_value(value) -> float:
    """
    Clean output value - convert torch tensors and numpy arrays to Python floats
    
    Parameters
    ----------
    value : any
        Output value from model
    
    Returns
    -------
    float
        Cleaned output value
    """
    if torch is not None and isinstance(value, torch.Tensor):
        return float(value.detach().cpu().numpy())
    elif np is not None and isinstance(value, np.ndarray):
        return float(value)
    else:
        return float(value)


def prepare_batch_inputs(inputs_list: List[Dict[str, float]], model: TorchModel) -> Dict[str, Any]:
    """
    Convert list of input dictionaries to batched tensors

    Parameters
    ----------
    inputs_list : List[Dict[str, float]]
        List of input dictionaries
    model : TorchModel
        The model (used to get input names and defaults)

    Returns
    -------
    Dict[str, Any]
        Dictionary with batched tensors/arrays for each input
    """
    if torch is None:
        raise ImportError("torch is required for batch processing")

    batch_size = len(inputs_list)
    batch_inputs = {}

    # For each input variable in the model
    for input_name in model.input_names:
        values = []
        for inputs in inputs_list:
            # Use provided value or model default
            if input_name in inputs:
                values.append(inputs[input_name])
            else:
                # Get default value from model
                default_val = None
                for var in model.input_variables:
                    if var.name == input_name:
                        default_val = var.default_value
                        break
                if default_val is not None:
                    values.append(default_val)
                else:
                    raise ValueError(f"No value or default found for input '{input_name}'")

        # Stack values into a batch tensor
        batch_inputs[input_name] = torch.tensor(values, dtype=torch.float32)

    return batch_inputs


def split_batch_outputs(batch_outputs: Dict[str, Any], batch_size: int) -> List[Dict[str, float]]:
    """
    Split batched outputs back into individual result dictionaries

    Parameters
    ----------
    batch_outputs : Dict[str, Any]
        Dictionary of batched output tensors/arrays
    batch_size : int
        Number of samples in the batch

    Returns
    -------
    List[Dict[str, float]]
        List of output dictionaries (one per sample)
    """
    results = []

    for i in range(batch_size):
        result = {}
        for key, value in batch_outputs.items():
            # Handle both tensors and arrays
            if torch is not None and isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    # Scalar tensor (batch size 1)
                    result[key] = clean_output_value(value)
                else:
                    # Batched tensor
                    result[key] = clean_output_value(value[i])
            elif np is not None and isinstance(value, np.ndarray):
                if value.ndim == 0:
                    result[key] = clean_output_value(value)
                else:
                    result[key] = clean_output_value(value[i])
            else:
                # Try indexing, fallback to the value itself
                try:
                    result[key] = clean_output_value(value[i])
                except (TypeError, IndexError):
                    result[key] = clean_output_value(value)
        results.append(result)

    return results


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    if DEFAULT_MODEL_NAME:
        try:
            # load model once
            model = load_lume_model(DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION)
            # store in app
            app.state.model = model
            app.state.model_name = DEFAULT_MODEL_NAME
            app.state.model_version = DEFAULT_MODEL_VERSION
            app.state.run_id = getattr(model, '_run_id', None)
            app.state.config_path = getattr(model, '_config_path', None)
            logger.info(" Model loaded and stored in app state")
        except Exception as e:
            logger.warning(f"Could not load default model on startup: {str(e)}")
            raise # Fail startup if model doesn't load 
    else:
        logger.warning("No default model specified via MODEL_NAME environment variable")
        raise ValueError("MODEL_NAME environment variable is required")
    
    yield
    
    # Shutdown (cleanup if needed)
    logger.info("Shutting down inference service")

    # Clean up downloaded artifacts if they exist
    if hasattr(app.state, 'config_path') and app.state.config_path:
        try:
            import shutil
            artifact_dir = Path(app.state.config_path).parent
            # Only remove if it's in a temp directory
            if str(artifact_dir).startswith(tempfile.gettempdir()):
                logger.info(f"Cleaning up temporary artifacts at: {artifact_dir}")
                shutil.rmtree(artifact_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Failed to clean up artifacts: {str(e)}")


# Create FastAPI app with lifespan
app = FastAPI(title="LUME Model Inference Service", lifespan=lifespan)


# Dependency: Get model from app state
async def get_model(request: Request) -> TorchModel:
    """Dependency injection to retrieve immutable model from app state"""
    if not hasattr(request.app.state, 'model') or request.app.state.model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    return request.app.state.model


@app.get("/")
async def root(request: Request):
    """Root endpoint with service info"""
    model_loaded = hasattr(request.app.state, 'model') and request.app.state.model is not None
    
    return {
        "service": "LUME Model Inference Service",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_loaded": model_loaded,
        "current_model": {
            "name": getattr(request.app.state, 'model_name', None),
            "version": getattr(request.app.state, 'model_version', None)
        } if model_loaded else None,
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "model_inputs": "/inputs",
            "model_outputs": "/outputs",
            "predict": "/predict",
            "predict_batch": "/predict/batch"
        }
    }

@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint for Kubernetes probes"""
    model_loaded = hasattr(request.app.state, 'model') and request.app.state.model is not None
    
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "model_name": getattr(request.app.state, 'model_name', None),
        "model_version": getattr(request.app.state, 'model_version', None)
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info(request: Request):
    """Get information about the currently loaded model"""
    if not hasattr(request.app.state, 'model') or request.app.state.model is None:
        return ModelInfo(loaded=False)
    
    model = request.app.state.model
    
    return ModelInfo(
        loaded=True,
        model_name=getattr(request.app.state, 'model_name', None),
        model_version=getattr(request.app.state, 'model_version', None),
        run_id=getattr(request.app.state, 'run_id', None),
        input_names=model.input_names,
        output_names=model.output_names
    )

@app.get("/inputs", response_model=ModelInputsResponse)
async def get_model_inputs(model: TorchModel = Depends(get_model)):
    """Get information about model input variables"""
    input_variables = {}
    for var in model.input_variables:
        input_variables[var.name] = {
            "default": var.default_value,
            "range": list(var.value_range) if var.value_range else None,
            "is_constant": var.is_constant,
            "unit": var.unit,
        }
    
    return ModelInputsResponse(
        input_names=model.input_names,
        input_variables=input_variables
    )

@app.get("/inputs/types", response_model=VariableTypesResponse)
async def get_variable_types(model: TorchModel = Depends(get_model)):
    """Get the types of all input and output variables"""
    input_types = {}
    for var in model.input_variables:
        input_types[var.name] = var.__class__.__name__
    
    output_types = {}
    for var in model.output_variables:
        output_types[var.name] = var.__class__.__name__
    
    return VariableTypesResponse(
        input_types=input_types,
        output_types=output_types
    )

@app.get("/outputs", response_model=ModelOutputsResponse)
async def get_model_outputs(model: TorchModel = Depends(get_model)):
    """Get information about model output variables"""
    output_variables = {}
    for var in model.output_variables:
        output_variables[var.name] = {
            "unit": var.unit,
        }
    
    return ModelOutputsResponse(
        output_names=model.output_names,
        output_variables=output_variables
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model: TorchModel = Depends(get_model)
):
    """
    Run model inference using LUME model.evaluate()
    
    Takes a dictionary of inputs and returns model predictions.
    Model is injected via dependency injection (thread-safe, immutable).
    """
    try:
        logger.debug(f"Received prediction request: {request.inputs}")
        
        # Evaluate the LUME model with timeout protection (30 seconds)
        try:
            outputs = await asyncio.wait_for(
                asyncio.to_thread(model.evaluate, request.inputs),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Prediction timeout after 30 seconds")
            raise HTTPException(status_code=504, detail="Prediction request timed out after 30 seconds")

        logger.debug(f"Raw model output: {outputs}")
        
        # Clean outputs
        cleaned_outputs = {k: clean_output_value(v) for k, v in outputs.items()}
        
        logger.debug(f"Prediction result: {cleaned_outputs}")
        
        return PredictionResponse(outputs=cleaned_outputs)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")



@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    request: BatchPredictionRequest,
    model: TorchModel = Depends(get_model)
):
    """
    Run batch inference - multiple predictions at once
    
    Handles both:
    1. Vectorized batch processing (passes all samples as batched tensors)
    2. Sequential processing (falls back if batch processing fails)

    Takes a list of input dictionaries and returns a list of predictions.
    Supports partial inputs (model will use defaults for missing values).
    Model is injected via dependency injection (thread-safe, immutable).
    """
    try:
        logger.debug(f"Received batch prediction request with {len(request.inputs_list)} samples")
        
        # Try vectorized batch processing first (more efficient)
        try:
            logger.debug("Attempting vectorized batch processing...")

            # Prepare batched tensor inputs
            batch_inputs = prepare_batch_inputs(request.inputs_list, model)
            logger.debug(f"Prepared batch inputs with shapes: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in batch_inputs.items()]}")

            # Single batched evaluation call with timeout
            batch_outputs = await asyncio.wait_for(
                asyncio.to_thread(model.evaluate, batch_inputs),
                timeout=30.0
            )

            logger.debug(f"Batch outputs: {[(k, v.shape if hasattr(v, 'shape') else type(v)) for k, v in batch_outputs.items()]}")

            # Split batch outputs back into individual results
            outputs_list = split_batch_outputs(batch_outputs, len(request.inputs_list))

            logger.info(f"✓ Vectorized batch processing succeeded for {len(outputs_list)} samples")

        except Exception as batch_error:
            # Fall back to sequential processing
            logger.warning(f"Vectorized batch processing failed ({str(batch_error)}), falling back to sequential processing")

            outputs_list = []

            for idx, inputs in enumerate(request.inputs_list):
                logger.debug(f"Processing sample {idx + 1}/{len(request.inputs_list)}: {inputs}")

                # Evaluate the LUME model with timeout protection
                try:
                    outputs = await asyncio.wait_for(
                        asyncio.to_thread(model.evaluate, inputs),
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Prediction timeout for sample {idx + 1} after 30 seconds")
                    raise HTTPException(
                        status_code=504,
                        detail=f"Prediction request timed out for sample {idx + 1} after 30 seconds"
                    )

                # Clean outputs
                cleaned_outputs = {k: clean_output_value(v) for k, v in outputs.items()}
                outputs_list.append(cleaned_outputs)

            logger.info(f"✓ Sequential processing completed for {len(outputs_list)} samples")

        logger.debug(f"Batch prediction completed: {len(outputs_list)} results")
        
        return BatchPredictionResponse(
            outputs_list=outputs_list,
            batch_size=len(outputs_list)
        )
    
    except asyncio.TimeoutError:
        logger.error(f"Batch prediction timeout after 30 seconds")
        raise HTTPException(status_code=504, detail="Batch prediction request timed out after 30 seconds")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/debug/state")
async def debug_state(request: Request):
    """Debug endpoint to see what's in app.state"""
    return {
        "has_model": hasattr(request.app.state, 'model'),
        "model_is_none": getattr(request.app.state, 'model', None) is None,
        "has_model_name": hasattr(request.app.state, 'model_name'),
        "model_name": getattr(request.app.state, 'model_name', None),
        "has_model_version": hasattr(request.app.state, 'model_version'),
        "model_version": getattr(request.app.state, 'model_version', None),
        "state_attributes": dir(request.app.state)
    }


if __name__ == "__main__":
    # Run the service
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )