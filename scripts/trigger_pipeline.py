import requests
import sys

try:
    url = "http://localhost:8000/run?dataset=SAMLUR_450163"
    print(f"Triggering pipeline at {url}...")
    resp = requests.post(url)
    print(f"Response ({resp.status_code}): {resp.json()}")
except Exception as e:
    print(f"Failed to trigger pipeline: {e}")
