import requests
import os
import sys

def download_weights():
    url = "https://ml-assets.apple.com/ml-depth-pro/weights/ml-depth-pro.pt"
    dest = "models/ml-depth-pro.pt"
    os.makedirs("models", exist_ok=True)
    
    print(f"Downloading {url} to {dest}...")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            done = int(50 * downloaded / total_size)
                            sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB")
                            sys.stdout.flush()
        print("\nDownload complete.")
    except Exception as e:
        print(f"\nError downloading weights: {e}")

if __name__ == "__main__":
    download_weights()
