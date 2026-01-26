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
    print(f"  {title}")


def compare_outputs(actual: dict, expected: dict, tolerance: float = 1e-4) -> bool:
    """
    Compare actual outputs with expected outputs
    
    Args:
        actual: Actual prediction outputs
        expected: Expected outputs
        tolerance: Allowed difference
    
    Returns:
        True if all outputs match within tolerance
    """
    all_match = True
    
    for key, expected_val in expected.items():
        if key not in actual:
            print(f"  Missing output: {key}")
            all_match = False
            continue
        
        actual_val = actual[key]
        diff = abs(actual_val - expected_val)
        
        if diff > tolerance:
            print(f"  {key}: expected {expected_val:.6f}, got {actual_val:.6f} (diff: {diff:.6e})")
            all_match = False
        else:
            print(f"  {key}: {actual_val:.6f} (expected {expected_val:.6f})")
    
    return all_match


def main():
    service_url = os.environ.get("INFERENCE_SERVICE_URL", "http://inference-service:8000")
    
    print_section("LUME MODEL VALIDATION TESTS")
    print(f"Service URL: {service_url}\n")
    
    client = InferenceClient(service_url)
    
    # Health check
    print_section("Health Check")
    if not client.health_check():
        print("Service is not healthy")
        sys.exit(1)
    print("Service is healthy")
    
    # Get model info and inputs
    model_info = client.get_model_info()
    inputs_info = client.get_inputs()
    
    print(f"\nModel: {model_info['model_name']} v{model_info['model_version']}")
    
    # Test 1: All inputs at default values
    print_section("Test 1: All Inputs at Default Values")
    
    all_defaults = {}
    for name, details in inputs_info['input_variables'].items():
        all_defaults[name] = details['default']
    
    print("Input (all defaults):")
    for name, val in list(all_defaults.items())[:3]:
        print(f"  {name}: {val}")
    print(f"  ... and {len(all_defaults) - 3} more\n")
    
    result = client.predict(all_defaults)
    
    expected_outputs = {
        'OTRS:IN20:571:XRMS': 304.6202,
        'OTRS:IN20:571:YRMS': 124.3262,
        'norm_emit_x': 5.6198e-07,
        'norm_emit_y': 5.6114e-07,
        'sigma_z': 0.0005
    }
    
    print("Comparing outputs:")
    test1_passed = compare_outputs(result['outputs'], expected_outputs, tolerance=0.001)
    
    if test1_passed:
        print("\n Test 1 PASSED")
    else:
        print("\n Test 1 FAILED")
    
    # Test 2: Only first 3 inputs (partial inputs)
    print_section("Test 2: Partial Inputs (First 3 Only)")
    
    partial_inputs = {}
    for i, (name, details) in enumerate(inputs_info['input_variables'].items()):
        if i < 3:
            partial_inputs[name] = details['default']
    
    print("Input (only 3 inputs, rest use model defaults):")
    for name, val in partial_inputs.items():
        print(f"  {name}: {val}")
    
    result = client.predict(partial_inputs)
    
    # Expected outputs are same as Test 1
    print("\nComparing outputs:")
    test2_passed = compare_outputs(result['outputs'], expected_outputs, tolerance=0.001)
    
    if test2_passed:
        print("\n Test 2 PASSED")
    else:
        print("\n Test 2 FAILED")
    
    # Test 3: Batch prediction for one input
    print_section("Test 3: Batch Prediction (One Input, 3 Values)")
    
    batch_inputs = [
        {"SOLN:IN20:121:BACT": 0.38},
        {"SOLN:IN20:121:BACT": 0.40},
        {"SOLN:IN20:121:BACT": 0.44}
    ]
    
    print("Batch inputs:")
    for i, inp in enumerate(batch_inputs):
        print(f"  Sample {i+1}: {inp}")
    
    batch_result = client.predict_batch(batch_inputs)
    
    print(f"\nReceived {batch_result['batch_size']} predictions\n")
    
    expected_batch_outputs = [
        {
            'OTRS:IN20:571:XRMS': 2504.2827,
            'OTRS:IN20:571:YRMS': 1050.7192,
            'norm_emit_x': 2.9335e-06,
            'norm_emit_y': 2.5953e-06,
            'sigma_z': 0.0004
        },
        {
            'OTRS:IN20:571:XRMS': 2268.1758,
            'OTRS:IN20:571:YRMS': 792.3976,
            'norm_emit_x': 2.5279e-06,
            'norm_emit_y': 2.5373e-06,
            'sigma_z': 0.0004
        },
        {
            'OTRS:IN20:571:XRMS': 1703.8993,
            'OTRS:IN20:571:YRMS': 411.8294,
            'norm_emit_x': 1.6844e-06,
            'norm_emit_y': 1.7893e-06,
            'sigma_z': 0.0004
        }
    ]
    
    test3_passed = True
    for i, (actual, expected) in enumerate(zip(batch_result['outputs_list'], expected_batch_outputs)):
        print(f"Sample {i+1}:")
        sample_passed = compare_outputs(actual, expected, tolerance=0.001)
        test3_passed = test3_passed and sample_passed
        print()
    
    if test3_passed:
        print(" Test 3 PASSED")
    else:
        print(" Test 3 FAILED")
    
    # Summary
    print_section("VALIDATION SUMMARY")
    print(f"Test 1 (All defaults): {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"Test 2 (Partial inputs): {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print(f"Test 3 (Batch prediction): {'✓ PASSED' if test3_passed else '✗ FAILED'}")
    
    all_passed = test1_passed and test2_passed and test3_passed
    
    if all_passed:
        print("\n ALL VALIDATION TESTS PASSED! \n")
        sys.exit(0)
    else:
        print("\n  SOME TESTS FAILED \n")
        sys.exit(1)


if __name__ == "__main__":
    main()