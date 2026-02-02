from client import InferenceClient

# Connect to local service
client = InferenceClient("http://localhost:8000")

# Test 1: Health
print("Test 1: Health Check")
is_healthy = client.health_check()
print(f"Healthy: {is_healthy}\n")

# Test 2: Get inputs to find defaults
print("Test 2: Get Model Inputs")
inputs_info = client.get_inputs()
print(f"Found {len(inputs_info['input_names'])} inputs\n")

# Build default inputs
all_defaults = {name: details['default'] 
                for name, details in inputs_info['input_variables'].items()}

print("First 3 defaults:")
for i, (name, val) in enumerate(list(all_defaults.items())[:3]):
    print(f"  {name}: {val}")
print()

# Test 3: Single prediction with all defaults
print("Test 3: Single Prediction (All Defaults)")
result = client.predict(all_defaults)
print("Outputs:")
for name, val in result['outputs'].items():
    print(f"  {name}: {val}")
print()

# Test 4: Partial inputs
print("Test 4: Partial Inputs (First 3 Only)")
partial_inputs = {name: val for name, val in list(all_defaults.items())[:3]}
print("Inputs:")
for name, val in partial_inputs.items():
    print(f"  {name}: {val}")

result = client.predict(partial_inputs)
print("\nOutputs:")
for name, val in result['outputs'].items():
    print(f"  {name}: {val}")
print()

# Test 5: Batch prediction
print("Test 5: Batch Prediction")
batch_inputs = [
    {"SOLN:IN20:121:BACT": 0.38},
    {"SOLN:IN20:121:BACT": 0.40},
    {"SOLN:IN20:121:BACT": 0.44}
]

print("Batch inputs:")
for i, inp in enumerate(batch_inputs):
    print(f"  Sample {i+1}: {inp}")

batch_result = client.predict_batch(batch_inputs)
print(f"\nGot {batch_result['batch_size']} predictions:\n")

for i, outputs in enumerate(batch_result['outputs_list']):
    print(f"Sample {i+1} outputs:")
    for name, val in outputs.items():
        print(f"  {name}: {val}")
    print()

print("All tests completed!")
