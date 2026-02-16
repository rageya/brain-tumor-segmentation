import numpy as np
import nibabel as nib
from pathlib import Path

def preprocess_input(file_path, target_shape=(128, 128, 128)):
    """Load and preprocess a NIfTI file"""
    img = nib.load(str(file_path)).get_fdata().astype(np.float32)
    
    if img.shape != target_shape:
        from scipy.ndimage import zoom
        factors = [t/s for t, s in zip(target_shape, img.shape)]
        img = zoom(img, factors, order=1)
    
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def load_multimodal_scan(t1_path, t1ce_path, t2_path, flair_path):
    """Load all 4 MRI modalities and stack them"""
    channels = []
    for path in [t1_path, t1ce_path, t2_path, flair_path]:
        if path and Path(path).exists():
            img = preprocess_input(path)
        else:
            img = np.zeros((128, 128, 128), dtype=np.float32)
        channels.append(img)
    
    return np.stack(channels, axis=0)

def calculate_tumor_metrics(pred_mask):
    """Calculate tumor volume metrics"""
    return {
        'total_voxels': np.sum(pred_mask > 0),
        'necrotic_voxels': np.sum(pred_mask == 1),
        'edema_voxels': np.sum(pred_mask == 2),
        'enhancing_voxels': np.sum(pred_mask == 3),
        'volume_ml': np.sum(pred_mask > 0) * 0.001
    }
