import requests
import os
import sys

def download_geosam():
    # Meta AI Segment Anything ViT-H Weights
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    dest = "models/sam_vit_h.pth"
    os.makedirs("models", exist_ok=True)
    
    print(f"Downloading GeoSAM (ViT-H) from {url}...")
    print(f"Destination: {os.path.abspath(dest)}")
    
    try:
        # No special headers needed for Meta public files usually, but good to have
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        with requests.get(url, headers=headers, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            done = int(50 * downloaded / total_size)
                            # Overwrite line for clean progress
                            sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB")
                            sys.stdout.flush()
        print("\nGeoSAM Download Complete.")
    except Exception as e:
        print(f"\nError downloading GeoSAM: {e}")

if __name__ == "__main__":
    download_geosam()
