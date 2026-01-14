import os
import logging
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import mlflow
from mlflow.tracking import MlflowClient
from lume_model.models import TorchModel

# Set up logging
logging.basicConfig(
    stream=sys.stdout,
    format="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="LUME Model Inference Service")

# Model configuration
DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", None)
DEFAULT_MODEL_VERSION = os.environ.get("MODEL_VERSION", None)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global variables
model: Optional[TorchModel] = None
current_model_name = None
current_model_version = None
model_config_path = None


# Request/Response models
class PredictionRequest(BaseModel):
    inputs: Dict[str, float]


class PredictionResponse(BaseModel):
    outputs: Dict[str, float]


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
    
    # Download artifacts to a temporary directory
    artifact_path = mlflow.artifacts.download_artifacts(run_id=run_id)
    
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
    
    # Common names for LUME model configs
    possible_names = [
        "model_config.yaml",
        "model.yaml",
        "config.yaml",
        "lume_model.yaml"
    ]
    
    # Search for YAML files
    for name in possible_names:
        yaml_path = artifact_dir / name
        if yaml_path.exists():
            logger.info(f"Found config file: {yaml_path}")
            return str(yaml_path)
    
    # If not found by name, search for any .yaml or .yml file
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


def load_lume_model(model_name: str, model_version: Optional[str] = None):
    """
    Load a LUME TorchModel from MLflow artifacts
    
    Parameters
    ----------
    model_name : str
        Name of the registered model in MLflow
    model_version : str, optional
        Version number, stage name, or "latest"
    """
    global model, current_model_name, current_model_version, model_config_path
    
    try:
        logger.info(f"Loading LUME model '{model_name}' version '{model_version}'...")
        
        # Download artifacts from MLflow
        run_id, artifact_path = download_model_artifacts(model_name, model_version)
        
        # Find the YAML config file
        yaml_config_path = find_yaml_config(artifact_path)
        
        logger.info(f"Loading TorchModel from: {yaml_config_path}")
        
        # Load the LUME TorchModel
        model = TorchModel(yaml_config_path)
        
        current_model_name = model_name
        current_model_version = model_version
        model_config_path = yaml_config_path

        model._run_id = run_id
        
        logger.info(f"✓ LUME model loaded successfully!")
        logger.info(f"  Model name: {current_model_name}")
        logger.info(f"  Model version: {current_model_version}")
        logger.info(f"  Run ID: {run_id}")
        logger.info(f"  Config path: {model_config_path}")
        logger.info(f"  Input variables: {model.input_names}")
        logger.info(f"  Output variables: {model.output_names}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}", exc_info=True)
        raise


@app.on_event("startup")
async def startup():
    """Load default model on startup if specified"""
    logger.info(f"MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
    
    if DEFAULT_MODEL_NAME:
        try:
            load_lume_model(DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION)
        except Exception as e:
            logger.warning(f"Could not load default model on startup: {str(e)}")
            logger.warning("Service will start without a loaded model. Use POST /model/load to load a model.")
    else:
        logger.info("No default model specified. Use POST /model/load to load a model.")


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "LUME Model Inference Service",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_loaded": model is not None,
        "current_model": {
            "name": current_model_name,
            "version": current_model_version
        } if model else None,
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "load_model": "POST /model/load",
            "model_inputs": "/inputs",
            "model_outputs": "/outputs",
            "predict": "/predict"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes probes"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": current_model_name,
        "model_version": current_model_version
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the currently loaded model"""
    if model is None:
        return ModelInfo(loaded=False)
    
    return ModelInfo(
        loaded=True,
        model_name=current_model_name,
        model_version=current_model_version,
         run_id=getattr(model, '_run_id', None),
        input_names=model.input_names,
        output_names=model.output_names
    )


@app.post("/model/load")
async def load_model_endpoint(request: LoadModelRequest):
    """
    Load a new LUME model from MLflow
    
    Downloads artifacts and loads the TorchModel from YAML config.
    """
    try:
        load_lume_model(request.model_name, request.model_version)
        
        return {
            "status": "success",
            "message": f"Model '{request.model_name}' version '{request.model_version}' loaded successfully",
            "model_name": current_model_name,
            "model_version": current_model_version,
            "config_path": model_config_path,
            "input_names": model.input_names,
            "output_names": model.output_names
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.get("/inputs", response_model=ModelInputsResponse)
async def get_model_inputs():
    """Get information about model input variables"""
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="No model loaded. Use POST /model/load to load a model first."
        )
    
    # Extract input variable info from LUME model
    # input_variables is a LIST, not a dict!
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

@app.get("/outputs", response_model=ModelOutputsResponse)
async def get_model_outputs():
    """Get information about model output variables"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Use POST /model/load to load a model first."
        )

    # Extract output variable info from LUME model
    # output_variables is a LIST, not a dict!
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
async def predict(request: PredictionRequest):
    """
    Run model inference using LUME model.evaluate()
    
    Takes a dictionary of inputs and returns model predictions.
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="No model loaded. Use POST /model/load to load a model first."
        )
    
    try:
        logger.debug(f"Received prediction request: {request.inputs}")
        
        # Evaluate the LUME model
        outputs = model.evaluate(request.inputs)
        
        logger.debug(f"Raw model output: {outputs}")
        
        # Clean outputs (convert torch tensors, numpy arrays to Python floats)
        cleaned_outputs = {}
        for k, v in outputs.items():
            try:
                import torch
                if isinstance(v, torch.Tensor):
                    cleaned_outputs[k] = float(v.detach().cpu().numpy())
                else:
                    cleaned_outputs[k] = float(v)
            except (ImportError, AttributeError):
                try:
                    import numpy as np
                    if isinstance(v, np.ndarray):
                        cleaned_outputs[k] = float(v)
                    else:
                        cleaned_outputs[k] = float(v)
                except ImportError:
                    cleaned_outputs[k] = float(v)
        
        logger.debug(f"Prediction result: {cleaned_outputs}")
        
        return PredictionResponse(outputs=cleaned_outputs)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    # Run the service
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
