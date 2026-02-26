#!/usr/bin/env python3
"""
Stress test script for inference service.

Continuously sends random inputs to the inference service and logs results.
Can run multiple instances simultaneously to simulate multiple GUIs.
"""

import requests
import random
import time
import argparse
import sys
from datetime import datetime
import statistics

# Configuration
INFERENCE_URL = "https://ard-modeling-service.slac.stanford.edu/cuinj"

# Input ranges for the two variables we're varying
INPUTS_CONFIG = {
    "QUAD:IN20:121:BACT": {
        "range": [-0.02098429469554406, 0.020999198106589838],
        "default": -0.001499227120199691
    },
    "QUAD:IN20:122:BACT": {
        "range": [-0.020998830517503037, 0.020998929132148195],
        "default": -0.0006872989433749197
    }
}

# Default values for all other inputs (from model)
DEFAULT_INPUTS = {
    "CAMR:IN20:186:R_DIST": 423.867825,
    "Pulse_length": 1.8550514181818183,
    "FBCK:BCI0:1:CHRG_S": 0.25,
    "SOLN:IN20:121:BACT": 0.4779693455075814,
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

class InferenceStressTester:
    """Stress tester for inference service"""
    def __init__(self, base_url, client_id=1, rate_hz=1.0):
        """
        Initialize stress tester.
        
        Parameters
        ----------
        base_url : str
            Base URL of inference service
        client_id : int
            Unique identifier for this client (for logging)
        rate_hz : float
            Request rate in Hz (requests per second)
        """
        self.base_url = base_url.rstrip('/')
        self.client_id = client_id
        self.rate_hz = rate_hz
        self.interval = 1.0 / rate_hz
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.response_times = []
        
        self.session = requests.Session()

    def generate_random_inputs(self):
        """
        Generate random inputs with varying QUAD values.
        
        Returns
        -------
        dict
            Dictionary of input values
        """
        inputs = {}
        
        # Add the two varying inputs with random values
        for var_name, config in INPUTS_CONFIG.items():
            min_val, max_val = config["range"]
            inputs[var_name] = random.uniform(min_val, max_val)
        
        return inputs
    
    def call_inference(self, inputs):
        """
        Call the inference service.
        
        Parameters
        ----------
        inputs : dict
            Input values
        
        Returns
        -------
        tuple
            (success: bool, response_time: float, outputs: dict or None, error: str or None)
        """
        start_time = time.time()
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json={"inputs": inputs},
                timeout=10
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                outputs = response.json()["outputs"]
                return True, response_time, outputs, None
            else:
                return False, response_time, None, f"HTTP {response.status_code}"
        
        except requests.Timeout:
            response_time = time.time() - start_time
            return False, response_time, None, "Timeout"
        
        except Exception as e:
            response_time = time.time() - start_time
            return False, response_time, None, str(e)
    
    def run_once(self):
        """
        Run one iteration: generate inputs, call inference, log result.
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        # Generate random inputs
        inputs = self.generate_random_inputs()
        
        # Call inference
        success, response_time, outputs, error = self.call_inference(inputs)
        
        # Update statistics
        self.total_requests += 1
        self.response_times.append(response_time)
        
        if success:
            self.successful_requests += 1
            
            # Log success
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[Client {self.client_id}] [{timestamp}] "
                  f" Request #{self.total_requests} | "
                  f"Time: {response_time*1000:.1f}ms")
            print(f"    Inputs:  QUAD:121={inputs['QUAD:IN20:121:BACT']:+.6f} | "
                  f"QUAD:122={inputs['QUAD:IN20:122:BACT']:+.6f}")
            print(f"    Outputs: XRMS={outputs.get('OTRS:IN20:571:XRMS', 0):.2f} | "
                  f"YRMS={outputs.get('OTRS:IN20:571:YRMS', 0):.2f} | "
                  f"sigma_z={outputs.get('sigma_z', 0):.6f} | "
                  f"emit_x={outputs.get('norm_emit_x', 0):.3e} | "
                  f"emit_y={outputs.get('norm_emit_y', 0):.3e}")
            
            return True
        
        else:
            self.failed_requests += 1
            
            # Log failure
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[Client {self.client_id}] [{timestamp}] "
                  f"✗ Request #{self.total_requests} | "
                  f"Error: {error} | "
                  f"Time: {response_time*1000:.1f}ms")
            
            return False
        
    def run(self, duration_seconds=None, num_requests=None):
        """
        Run stress test continuously.
        
        Parameters
        ----------
        duration_seconds : int, optional
            Run for specified duration in seconds. If None, run forever.
        num_requests : int, optional
            Run for specified number of requests. If None, run forever.
        """
        print(f"\n{'='*80}")
        print(f"Starting Stress Test - Client {self.client_id}")
        print(f"{'='*80}")
        print(f"Inference URL: {self.base_url}")
        print(f"Request rate: {self.rate_hz} Hz ({self.interval:.2f}s interval)")
        if duration_seconds:
            print(f"Duration: {duration_seconds} seconds")
        if num_requests:
            print(f"Number of requests: {num_requests}")
        print(f"Varying inputs: {', '.join(INPUTS_CONFIG.keys())}")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        try:
            while True:
                # Check stopping conditions
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    break
                if num_requests and self.total_requests >= num_requests:
                    break
                
                # Run one iteration
                iteration_start = time.time()
                self.run_once()
                
                # Sleep to maintain rate
                elapsed = time.time() - iteration_start
                sleep_time = max(0, self.interval - elapsed)
                time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print(f"\n[Client {self.client_id}] Interrupted by user")
        
        finally:
            self.print_summary()
    
    def print_summary(self):
        """Print test summary statistics"""
        print(f"Test Summary - Client {self.client_id}")
        print(f"Total requests:      {self.total_requests}")
        print(f"Successful:          {self.successful_requests} ({self.successful_requests/max(self.total_requests,1)*100:.1f}%)")
        print(f"Failed:              {self.failed_requests} ({self.failed_requests/max(self.total_requests,1)*100:.1f}%)")

        if self.response_times:
            print(f"\nResponse Times:")
            print(f"  Min:     {min(self.response_times)*1000:.1f} ms")
            print(f"  Max:     {max(self.response_times)*1000:.1f} ms")
            print(f"  Mean:    {statistics.mean(self.response_times)*1000:.1f} ms")
            print(f"  Median:  {statistics.median(self.response_times)*1000:.1f} ms")
        
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Stress test inference service with random inputs"
    )
    parser.add_argument(
        "--url",
        default=INFERENCE_URL,
        help=f"Inference service URL (default: {INFERENCE_URL})"
    )
    parser.add_argument(
        "--client-id",
        type=int,
        default=1,
        help="Client ID for logging (default: 1)"
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=1.0,
        help="Request rate in Hz (default: 1.0)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Test duration in seconds (default: run forever)"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        help="Number of requests to make (default: run forever)"
    )
    
    args = parser.parse_args()
    
    # Create and run tester
    tester = InferenceStressTester(
        base_url=args.url,
        client_id=args.client_id,
        rate_hz=args.rate
    )
    
    tester.run(
        duration_seconds=args.duration,
        num_requests=args.num_requests
    )


if __name__ == "__main__":
    main()
        
    



    




