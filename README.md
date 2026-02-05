# Kubernetes Deployment Template for Inference Service

This template generates Kubernetes manifests for deploying ML inference services. User can either use a yaml like [here](https://github.com/slaclab/inference-service/blob/main/model-configs/cu-inj.yaml) or use copier.

The command below will create the deployment yaml in the folder specified at the end. 

## Usage

```bash
# Generate deployment for a new model
copier copy --data-file model-configs/iris-model.yaml copier-template-k8s deployments/iris-model


# Or from your project root
copier copy copier-template-k8s deployments/iris-model
```
## Example

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


## Testing
```bash
kubectl run test-model \
> --image=ghcr.io/slaclab/inference-service/test-client:latest \
> --rm -it --restart=Never \
> --env="INFERENCE_SERVICE_URL=http://<service name>:8000" \
> -n inference-service \
> python test_client.py

```