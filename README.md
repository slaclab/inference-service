## To deploy application

```bash
# Apply ConfigMap
kubectl apply -f k8s/configmap.yaml

# Apply Deployment
kubectl apply -f k8s/deployment.yaml

# Apply Service
kubectl apply -f k8s/service.yaml
```

## To test models

```bash
kubectl run test-client --image=curlimages/curl --rm -it --restart=Never -- sh
# Test health
curl http://inference-service:8000/health

# Test model info
curl http://inference-service:8000/model/info

# Test prediction (simple test)
curl -X POST http://inference-service:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"CAMR:IN20:186:R_DIST": 423.0, "Pulse_length": 1.85, "FBCK:BCI0:1:CHRG_S": 0.25, "SOLN:IN20:121:BACT": 0.47, "QUAD:IN20:121:BACT": -0.001, "QUAD:IN20:122:BACT": -0.0006, "ACCL:IN20:300:L0A_ADES": 58.0, "ACCL:IN20:300:L0A_PDES": -9.5, "ACCL:IN20:400:L0B_ADES": 70.0, "ACCL:IN20:400:L0B_PDES": 9.8, "QUAD:IN20:361:BACT": -2.0, "QUAD:IN20:371:BACT": 2.0, "QUAD:IN20:425:BACT": -1.08, "QUAD:IN20:441:BACT": -0.17, "QUAD:IN20:511:BACT": 2.85, "QUAD:IN20:525:BACT": -3.21}}'

# Exit
exit
```

## For client
Make sure this is deployed in the same namespace as the inference service.

```bash
# Deploy the job
kubectl apply -f k8s/test-client-job.yaml

# Watch the job
kubectl get jobs -w

# View logs
kubectl logs job/test-client

```