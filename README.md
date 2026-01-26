## To deploy application

```bash
chmod +x deploy.sh
,/deploy.sh
```

## To test models

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "CAMR:IN20:186:R_DIST": 423.867825,
      "Pulse_length": 1.8550514181818183,
      "FBCK:BCI0:1:CHRG_S": 0.25,
      "SOLN:IN20:121:BACT": 0.4779693455075814,
      "QUAD:IN20:121:BACT": -0.001499227120199691,
      "QUAD:IN20:122:BACT": -0.0006872989433749197,
      "ACCL:IN20:300:L0A_ADES": 58.0,
      "ACCL:IN20:300:L0A_PDES": -9.53597349,
      "ACCL:IN20:400:L0B_ADES": 70.0,
      "ACCL:IN20:400:L0B_PDES": 9.85566222,
      "QUAD:IN20:361:BACT": -2.0005920106399526,
      "QUAD:IN20:371:BACT": 2.0005920106399526,
      "QUAD:IN20:425:BACT": -1.0807627139393465,
      "QUAD:IN20:441:BACT": -0.17938799998564897,
      "QUAD:IN20:511:BACT": 2.852171999771826,
      "QUAD:IN20:525:BACT": -3.218399988942528
    }
  }' | python -m json.tool


curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "CAMR:IN20:186:R_DIST": 423.867825,
      "Pulse_length": 1.8550514181818183,
      "FBCK:BCI0:1:CHRG_S": 0.25
    }
  }' | python -m json.tool

curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "inputs_list": [
      {"SOLN:IN20:121:BACT": 0.38},
      {"SOLN:IN20:121:BACT": 0.40},
      {"SOLN:IN20:121:BACT": 0.44}
    ]
  }' | python -m json.tool


```