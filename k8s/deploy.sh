#!/bin/bash

set -e

# Configuration
NAMESPACE="inference-service"
MANIFEST="deployment.yaml"

echo "  Deploying Inference Service"


# Check if namespace exists, create if not
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    echo "Creating namespace: $NAMESPACE"
    kubectl create namespace $NAMESPACE
else
    echo "Using existing namespace: $NAMESPACE"
fi

# Apply the manifest
echo ""
echo "Applying Kubernetes manifests..."
kubectl apply -f $MANIFEST

# Wait for deployment
echo ""
echo "Waiting for inference service to be ready..."
kubectl rollout status deployment/inference-service -n $NAMESPACE --timeout=5m

# Check pods
echo ""
echo "Checking pod status..."
kubectl get pods -n $NAMESPACE -l app=inference-service

# Check service
echo ""
echo "Checking service..."
kubectl get svc inference-service -n $NAMESPACE

# Run validation test
echo ""
echo "  Running Validation Tests"


# Delete any existing validation job
kubectl delete job validation-test -n $NAMESPACE 2>/dev/null || true

# Wait a moment
sleep 2

# Apply validation job
kubectl apply -f $MANIFEST

# Wait for job to complete
echo ""
echo "Waiting for validation test to complete..."
kubectl wait --for=condition=complete --timeout=5m job/validation-test -n $NAMESPACE || true

# Show test results
echo ""
echo "  Validation Test Results"
kubectl logs job/validation-test -n $NAMESPACE

# Check if tests passed
if kubectl logs job/validation-test -n $NAMESPACE | grep -q "ALL VALIDATION TESTS PASSED"; then
    echo ""
    echo "DEPLOYMENT SUCCESSFUL! All tests passed."
    echo ""
    echo "Service is available at: http://inference-service.$NAMESPACE.svc.cluster.local:8000"
    exit 0
else
    echo ""
    echo " DEPLOYMENT COMPLETED but tests failed. Check logs above."
    exit 1
fi