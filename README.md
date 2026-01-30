# Kubernetes Deployment Template for Inference Service

This template generates Kubernetes manifests for deploying ML inference services.

## Usage

```bash
# Generate deployment for a new model
copier copy --data-file model-configs/iris-model.yaml copier-template-k8s deployments/iris-model


# Or from your project root
copier copy copier-template-k8s deployments/iris-model
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