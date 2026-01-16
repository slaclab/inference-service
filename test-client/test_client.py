import os
import sys
import json
from client import InferenceClient
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def main():
    # Get service URL from environment variable
    service_url = os.environ.get("INFERENCE_SERVICE_URL", "http://inference-service:8000")
    
    print_section("INFERENCE SERVICE CLIENT TEST")
    print(f"Service URL: {service_url}\n")
    
    # Initialize client
    client = InferenceClient(service_url)
    
    # Test 1: Health Check
    print_section("1. Health Check")
    try:
        is_healthy = client.health_check()
        if is_healthy:
            print("✓ Service is healthy")
        else:
            print("✗ Service is not healthy")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        sys.exit(1)
    
    # Test 2: Get Model Info
    print_section("2. Model Info")
    try:
        model_info = client.get_model_info()
        print(json.dumps(model_info, indent=2))
        
        if not model_info.get("loaded"):
            print("\n✗ No model loaded!")
            sys.exit(1)
        
        print(f"\n✓ Model loaded: {model_info['model_name']} v{model_info['model_version']}")
    except Exception as e:
        print(f"✗ Failed to get model info: {e}")
        sys.exit(1)
    
    # Test 3: Get Inputs
    print_section("3. Model Inputs")
    try:
        inputs_info = client.get_inputs()
        print(f"Input names: {inputs_info['input_names']}\n")
        
        # Show first 3 inputs in detail
        for i, (name, details) in enumerate(list(inputs_info['input_variables'].items())[:3]):
            print(f"{name}:")
            print(f"  Default: {details.get('default')}")
            print(f"  Range: {details.get('range')}")
            print(f"  Unit: {details.get('unit')}")
            print()
        
        if len(inputs_info['input_variables']) > 3:
            print(f"... and {len(inputs_info['input_variables']) - 3} more inputs\n")
        
        print(f"✓ Got {len(inputs_info['input_names'])} inputs")
    except Exception as e:
        print(f"✗ Failed to get inputs: {e}")
        sys.exit(1)
    
    # Test 4: Get Outputs
    print_section("4. Model Outputs")
    try:
        outputs_info = client.get_outputs()
        print(f"Output names: {outputs_info['output_names']}\n")
        
        for name, details in outputs_info['output_variables'].items():
            print(f"{name}:")
            print(f"  Unit: {details.get('unit')}")
        
        print(f"\n✓ Got {len(outputs_info['output_names'])} outputs")
    except Exception as e:
        print(f"✗ Failed to get outputs: {e}")
        sys.exit(1)
    
    # Test 5: Make Prediction
    print_section("5. Prediction Test")
    try:
        # Create test inputs using default values
        test_inputs = {}
        for name, details in inputs_info['input_variables'].items():
            default_val = details.get('default')
            if default_val is not None:
                test_inputs[name] = default_val
            else:
                # Use middle of range if no default
                value_range = details.get('range')
                if value_range and len(value_range) == 2:
                    test_inputs[name] = (value_range[0] + value_range[1]) / 2
                else:
                    test_inputs[name] = 0.0
        
        print("Test inputs (using defaults):")
        for i, (name, value) in enumerate(list(test_inputs.items())[:5]):
            print(f"  {name}: {value}")
        if len(test_inputs) > 5:
            print(f"  ... and {len(test_inputs) - 5} more")
        
        print("\nCalling predict endpoint...")
        prediction = client.predict(test_inputs)
        
        print("\nPrediction outputs:")
        print(json.dumps(prediction['outputs'], indent=2))
        
        print(f"\n✓ Prediction successful!")
        print(f"  Got {len(prediction['outputs'])} outputs")
        
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    print_section("✓ ALL TESTS PASSED")
    print(f"Model: {model_info['model_name']} v{model_info['model_version']}")
    print(f"Inputs: {len(inputs_info['input_names'])}")
    print(f"Outputs: {len(outputs_info['output_names'])}")
    print(f"Service URL: {service_url}")
    print()


if __name__ == "__main__":
    main()