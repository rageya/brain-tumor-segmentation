from pathlib import Path

def get_examples():
    """
    Get example data paths for Gradio interface
    Updated to work with data/ folder structure
    """
    data_dir = Path("data")
    examples = []
    
    if not data_dir.exists():
        print("⚠️ No data folder found")
        return []
    
    # Find patient folders in data/
    patient_folders = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("BraTS")])
    
    print(f"🔍 Found {len(patient_folders)} patient folders in data/")
    
    for patient_dir in patient_folders[:3]:  # Use first 3 patients
        patient_name = patient_dir.name
        print(f"  📁 Checking {patient_name}...")
        
        # Look for MRI files
        t1 = patient_dir / f"{patient_name}_t1.nii.gz"
        t1ce = patient_dir / f"{patient_name}_t1ce.nii.gz"
        t2 = patient_dir / f"{patient_name}_t2.nii.gz"
        flair = patient_dir / f"{patient_name}_flair.nii.gz"
        
        # Check if all modalities exist
        if t1.exists() and t1ce.exists() and t2.exists() and flair.exists():
            examples.append([
                str(t1),
                str(t1ce),
                str(t2),
                str(flair)
            ])
            print(f"    ✅ All modalities found")
        else:
            missing = []
            if not t1.exists(): missing.append("t1")
            if not t1ce.exists(): missing.append("t1ce")
            if not t2.exists(): missing.append("t2")
            if not flair.exists(): missing.append("flair")
            print(f"    ⚠️ Missing: {', '.join(missing)}")
    
    if examples:
        print(f"✅ Loaded {len(examples)} example patients")
    else:
        print("⚠️ No valid patient examples found")
    
    return examples
