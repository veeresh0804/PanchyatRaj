import torch

def inspect_weights(path):
    print(f"Inspecting: {path}")
    try:
        sd = torch.load(path, map_location="cpu")
        keys = list(sd.keys())
        print(f"Total keys: {len(keys)}")
        print("First 10 keys:")
        for k in keys[:10]:
            print(f"  {k}: {sd[k].shape}")
        
        # Check first upblock weight to determine channel sizes
        up1_key = "up1.0.weight"
        if up1_key in sd:
            print(f"Layer {up1_key}: {sd[up1_key].shape}")
        
        encoder_keys = [k for k in keys if k.startswith("encoder.")]
        backbone_keys = [k for k in keys if k.startswith("backbone.")]
        print(f"Encoder keys: {len(encoder_keys)}")
        print(f"Backbone keys: {len(backbone_keys)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_weights("models/swin/swin_fold0_best.pth")
