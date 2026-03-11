import requests
import time
import sys

def run_benchmark(dataset_name="Benchmark_Village"):
    print(f"Starting Benchmark for dataset: {dataset_name}")
    start_time = time.time()
    
    # 1. Trigger the pipeline via API
    try:
        res = requests.post(f"http://localhost:8000/run?dataset={dataset_name}")
        if res.status_code != 200 or res.json().get("status") not in ["started", "running"]:
            print("Failed to dispatch pipeline.")
            return
        print("Pipeline dispatched to worker pool.")
    except Exception as e:
        print(f"API Error. Ensure Uvicorn is running: {e}")
        return

    # 2. Poll metrics directly
    max_wait = 600 # 10 minutes timeout
    tiles_total = 0
    
    while time.time() - start_time < max_wait:
        try:
            status_res = requests.get("http://localhost:8000/status").json()
            state = status_res.get("state", "UNKNOWN")
            tiles_done = status_res.get("tiles_done", 0)
            tiles_total = status_res.get("tiles_total", 1)
            
            elapsed = time.time() - start_time
            sys.stdout.write(f"\rElapsed: {elapsed:.1f}s | State: {state} | Tiles: {tiles_done}/{tiles_total}")
            sys.stdout.flush()
            
            if state in ["COMPLETED", "FAILED"]:
                print("\n\n--- BENCHMARK RESULTS ---")
                print(f"Total Runtime:     {elapsed:.2f} seconds")
                print(f"Total Tiles:       {tiles_total}")
                if elapsed > 0:
                    print(f"Throughput:        {(tiles_total / elapsed):.2f} tiles/sec")
                print(f"Final State:       {state}")
                if state == "COMPLETED" and elapsed < 180:
                    print("SUCCESS: Target constraint (<3 minutes) MET.")
                break
                
        except Exception as e:
            pass
            
        time.sleep(2)
        
if __name__ == "__main__":
    run_benchmark()
