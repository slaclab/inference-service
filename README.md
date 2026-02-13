# Kubernetes Deployment Template for Inference Service


This repo contains code for FastAPI service for serving LUME models from MLFlow. With one docker image, multiple model deployments can be made via environment variables. Any changes to the client code will rebuild the base image automatically through Github Actions CI.

### Environment Variables
1. MLFLOW_TRACKING_URI: This is set in the mlflow-config configmap. 
2. MODEL_NAME: lcls-cu-inj-model or lcls-fel-surrogate
3. MODEL_VERSION: 1

### Testing inference service image 
To test the functionality of the image, user can create a temporary pod using the test-client image (also created by the CI) to run checks.

```bash
kubectl run test -n inference-service --image=ghcr.io/slaclab/inference-service/test-client:latest --rm -it --restart=Never --env="INFERENCE_SERVICE_URL=http://inference-service:8000" python test_validation.py

kubectl run test -n inference-service --image=ghcr.io/slaclab/inference-service/test-client:latest --rm -it --restart=Never --env="INFERENCE_SERVICE_URL=http://inference-service:8000" python test_client.py
```

## Template for inference service deployment

The Copier template generates Kubernetes manifests for deploying ML inference services. User can either use a yaml like [here](https://github.com/slaclab/inference-service/blob/main/model-configs/cu-inj.yaml) or use copier.

1. User can either use a simple template that is then used by Copier to generate deployment yaml

```
service_name: iris-service
namespace: inference-service
model_name: iris-model
model_version: "1"
# mlflow_uri removed - it's in the shared mlflow-config ConfigMap
container_registry: ghcr.io/slaclab/inference-service
replicas: 2
memory_request: "4Gi"
memory_limit: "8Gi"
cpu_request: "1000m"
cpu_limit: "4000m"
```
An example of this template is in the model-configs/iris-model directory. User can generate copier template in the deployments/iris-model directory using this command from the root - 

```
copier copy --data-file model-configs/iris-model.yaml copier-template-k8s deployments/iris-model
```

2. User can also simply run below copier command and answer questions to generate the deployment yaml 

```
copier copy copier-template-k8s deployments/iris-model
```

The command below will create the deployment yaml in the folder specified at the end. These are the questions the template will ask.

```
(test-bed) bash-5.3$ copier copy copier-template-k8s deployments/fel-model
🎤 What is the service name?
   inference-service-fel
🎤 Which Kubernetes namespace?
   lume-online-ml
🎤 What is the MLflow model name?
   lcls-fel-surrogate
🎤 What model version to deploy?
   1
🎤 Container registry (e.g., ghcr.io/username/repo)?
   ghcr.io/slaclab/inference-service
🎤 Number of replicas?
   1
🎤 Memory request (e.g., 2Gi)?
   2Gi
🎤 Memory limit (e.g., 4Gi)?
   4Gi
🎤 CPU request (e.g., 500m)?
   500m
🎤 CPU limit (e.g., 2000m)?
   2000m

Copying from template version None
    create  deployment.yaml
```

User can then deploy the deployment yaml in the lume-online-ml namespace either manually using 
```kubectl apply -f deployment.yaml -n lume-online-ml```. We have also configured ArgoCD to automate the deployments. 