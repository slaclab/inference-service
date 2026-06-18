## Usage

Single client (1 request/second):
```python stress_test.py```

Faster rate (10 requests/second):
```python stress_test.py --rate 10```

Run for 60 seconds:
```python stress_test.py --duration 60```

Run exactly 100 requests:
```python stress_test.py --num-requests 100```

To run multiple clients at one time 
```bash
chmod +x run_multiple.sh

# 4 clients, 1 Hz each, 60 seconds
./run_multiple.sh 4 1.0 60

# View logs
tail -f client_*.log
```