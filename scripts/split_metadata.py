
import json
import os
from pathlib import Path

def split_metadata(meta_json, output_dir):
    with open(meta_json, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    count = 0
    for entry in data:
        tile_id = entry.get('tile_id')
        if not tile_id:
            continue
            
        # 1. Add prefix 'tile_' as expected by run_inference.py
        out_path = os.path.join(output_dir, f"tile_{tile_id}.json")
        
        # 2. Map 'bbox' to 'bounds' as expected by orchestrator.py and serve.py
        if 'bbox' in entry and 'bounds' not in entry:
            entry['bounds'] = entry['bbox']
            
        with open(out_path, 'w') as f_out:
            json.dump(entry, f_out, indent=2)
        count += 1
    
    print(f"Split {count} metadata entries into {output_dir} with 'tile_' prefix and 'bounds' mapping.")

def clean_old_jsons(output_dir):
    # Remove jsons without tile_ prefix
    count = 0
    for f in os.listdir(output_dir):
        if f.endswith(".json") and not f.startswith("tile_"):
            os.remove(os.path.join(output_dir, f))
            count += 1
    print(f"Cleaned {count} old JSON files.")

if __name__ == "__main__":
    meta_path = r"c:\Users\manoh\Desktop\Panchayat Raj\hydra-pramanam-master\data\clean_dataset_2\metadata\tiles.json"
    tiles_dir = r"c:\Users\manoh\Desktop\Panchayat Raj\hydra-pramanam-master\data\clean_dataset_2\tiles\512"
    clean_old_jsons(tiles_dir)
    split_metadata(meta_path, tiles_dir)
