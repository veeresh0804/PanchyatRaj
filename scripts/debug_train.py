"""Debug training setup."""
import sys, os, traceback
sys.path.insert(0, '.')

try:
    from src.utils.io import load_config
    config = load_config('config/small_config.yaml')
    from src.train.train_swin import collect_data, get_fold_splits
    tile_paths, mask_paths = collect_data(config)
    valid_masks = sum(1 for m in mask_paths if m and os.path.isfile(m))
    print(f"Tiles: {len(tile_paths)}, masks with files: {valid_masks}")
    train_t, train_m, val_t, val_m = get_fold_splits(tile_paths, mask_paths, 2, 0)
    print(f"Train: {len(train_t)}, Val: {len(val_t)}")
    
    import torch
    print(f"CUDA: {torch.cuda.is_available()}")
    swin_cfg = config['swin']
    print(f"Creating SwinUNet: encoder={swin_cfg['encoder']}, input_size={swin_cfg['input_size']}")
    
    from src.models.swin_unet import SwinUNet
    model = SwinUNet(
        encoder_name=swin_cfg['encoder'],
        pretrained=swin_cfg['pretrained'],
        num_classes=swin_cfg['num_classes'],
        input_size=swin_cfg['input_size'],
    )
    print('Model created successfully')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'Model on {device}')
    
    # Test forward pass
    x = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Forward pass OK: logits shape={out['logits'].shape}")
    
except Exception as e:
    traceback.print_exc()
