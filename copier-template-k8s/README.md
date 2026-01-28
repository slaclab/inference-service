# copier-template-k8s/README.md

# Kubernetes Deployment Template for Inference Service

This template generates Kubernetes manifests for deploying ML inference services.

## Usage

```bash
# Generate deployment for a new model
copier copy --data-file model-configs/iris-model.yaml copier-template-k8s deployments/iris-model


# Or from your project root
copier copy copier-template-k8s deployments/fel-model
```