import gradio as gr
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import io
from model import UNETRAdvanced, CONFIG
from examples import get_examples

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")

model = UNETRAdvanced(CONFIG).to(device)
print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Load weights
model_path = Path("best_safe_model.pth")
if model_path.exists():
    print(f"📦 Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    
    # Check for deep supervision heads
    ds_keys = [k for k in state_dict.keys() if k.startswith('ds_head')]
    if ds_keys:
        print(f"  Found {len(ds_keys)} deep supervision layers (used during training)")
        for key in ds_keys:
            del state_dict[key]
            print(f"    Removed: {key}")
    
    model.load_state_dict(state_dict, strict=False)
    print("✅ Model weights loaded successfully!")
else:
    alt_path = Path("best_model.pth")
    if alt_path.exists():
        print(f"📦 Loading from alternative path: {alt_path}")
        state_dict = torch.load(alt_path, map_location=device)
        ds_keys = [k for k in state_dict.keys() if k.startswith('ds_head')]
        for key in ds_keys:
            del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model loaded!")
    else:
        print("⚠️ WARNING: No trained weights found!")
        print("   Upload best_safe_model.pth to the Files tab")

model.eval()

# Global storage
current_volume = None
current_prediction = None

def find_best_slices(pred_mask):
    """Find best slice for each view"""
    # Axial (z-axis)
    axial_counts = np.sum(pred_mask > 0, axis=(0, 1))
    best_axial = int(np.argmax(axial_counts)) if np.max(axial_counts) > 0 else 64
    
    # Sagittal (x-axis)
    sagittal_counts = np.sum(pred_mask > 0, axis=(1, 2))
    best_sagittal = int(np.argmax(sagittal_counts)) if np.max(sagittal_counts) > 0 else 64
    
    # Coronal (y-axis)
    coronal_counts = np.sum(pred_mask > 0, axis=(0, 2))
    best_coronal = int(np.argmax(coronal_counts)) if np.max(coronal_counts) > 0 else 64
    
    return best_axial, best_sagittal, best_coronal

def create_slice_visualization(img_slice, mask_slice, slice_idx, view_name, total_slices):
    """Create visualization for a single slice"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original MRI
    axes[0].imshow(img_slice, cmap='gray', aspect='auto')
    axes[0].set_title(f"Original MRI\n{view_name} View - Slice {slice_idx}/{total_slices}", 
                     fontsize=13, color='white', weight='bold', pad=10)
    axes[0].axis('off')
    
    # Segmentation mask only
    mask_colored = np.zeros((*mask_slice.shape, 3))
    mask_colored[mask_slice == 1] = [1, 0, 0]  # Red: Necrotic
    mask_colored[mask_slice == 2] = [0, 1, 0]  # Green: Edema
    mask_colored[mask_slice == 3] = [0, 0, 1]  # Blue: Enhancing
    
    axes[1].imshow(mask_colored, aspect='auto')
    axes[1].set_title("Segmentation Mask", fontsize=13, color='cyan', weight='bold', pad=10)
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img_slice, cmap='gray', aspect='auto')
    tumor_count = np.sum(mask_slice > 0)
    
    if tumor_count > 0:
        overlay = np.zeros((*mask_slice.shape, 4))
        overlay[mask_slice == 1] = [1, 0, 0, 0.6]
        overlay[mask_slice == 2] = [0, 1, 0, 0.4]
        overlay[mask_slice == 3] = [0, 0, 1, 0.8]
        axes[2].imshow(overlay, aspect='auto')
    
    # Count pixels per class
    necrotic = np.sum(mask_slice == 1)
    edema = np.sum(mask_slice == 2)
    enhancing = np.sum(mask_slice == 3)
    
    title = f"Overlay ({tumor_count} tumor pixels)"
    if necrotic > 0 or edema > 0 or enhancing > 0:
        title += f"\n🔴{necrotic} 🟢{edema} 🔵{enhancing}"
    
    axes[2].set_title(title, fontsize=13, color='yellow', weight='bold', pad=10)
    axes[2].axis('off')
    
    fig.patch.set_facecolor('#1e1e1e')
    plt.tight_layout(pad=2.0)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#1e1e1e', dpi=120, pad_inches=0.2)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    
    return img

def visualize_axial(slice_idx):
    """Axial view (top-down)"""
    global current_volume, current_prediction
    if current_volume is None or current_prediction is None:
        return gr.update()
    
    slice_idx = int(slice_idx)
    img_slice = current_volume[1, :, :, slice_idx]  # T1ce
    mask_slice = current_prediction[:, :, slice_idx]
    
    return create_slice_visualization(img_slice, mask_slice, slice_idx, "Axial", 127)

def visualize_sagittal(slice_idx):
    """Sagittal view (side view)"""
    global current_volume, current_prediction
    if current_volume is None or current_prediction is None:
        return gr.update()
    
    slice_idx = int(slice_idx)
    img_slice = current_volume[1, slice_idx, :, :]  # T1ce
    mask_slice = current_prediction[slice_idx, :, :]
    
    return create_slice_visualization(img_slice, mask_slice, slice_idx, "Sagittal", 127)

def visualize_coronal(slice_idx):
    """Coronal view (front view)"""
    global current_volume, current_prediction
    if current_volume is None or current_prediction is None:
        return gr.update()
    
    slice_idx = int(slice_idx)
    img_slice = current_volume[1, :, slice_idx, :]  # T1ce
    mask_slice = current_prediction[:, slice_idx, :]
    
    return create_slice_visualization(img_slice, mask_slice, slice_idx, "Coronal", 127)

def predict_and_analyze(t1_file, t1ce_file, t2_file, flair_file):
    """Main prediction function with correct preprocessing"""
    global current_volume, current_prediction
    
    try:
        # Load modalities with EXACT training preprocessing
        channels = []
        file_names = ['T1', 'T1ce', 'T2', 'FLAIR']
        
        for idx, file in enumerate([t1_file, t1ce_file, t2_file, flair_file]):
            if file is None:
                print(f"⚠️ {file_names[idx]} not provided, using zeros")
                channels.append(np.zeros((128, 128, 128), dtype=np.float32))
            else:
                filepath = file if isinstance(file, str) else file.name
                print(f"📂 Loading {file_names[idx]}: {Path(filepath).name}")
                
                img = nib.load(filepath).get_fdata().astype(np.float32)
                print(f"  Original shape: {img.shape}, range: [{img.min():.2f}, {img.max():.2f}]")
                
                # Resize to 128x128x128 if needed
                if img.shape != (128, 128, 128):
                    from scipy.ndimage import zoom
                    factors = [128/s for s in img.shape]
                    img = zoom(img, factors, order=1)
                    print(f"  Resized to: {img.shape}")
                
                # ===== MATCH TRAINING NORMALIZATION =====
                # Remove outliers
                p1 = np.percentile(img, 1)
                p99 = np.percentile(img, 99)
                img = np.clip(img, p1, p99)
                
                # Z-score normalization per modality
                mean = img[img > 0].mean() if np.any(img > 0) else 0
                std = img[img > 0].std() if np.any(img > 0) else 1
                
                if std > 0:
                    img = (img - mean) / std
                else:
                    img = img - mean
                
                # Clip to reasonable range
                img = np.clip(img, -5, 5)
                
                print(f"  After norm: range: [{img.min():.2f}, {img.max():.2f}], mean: {img.mean():.2f}")
                channels.append(img)
        
        image = np.stack(channels, axis=0)
        current_volume = image
        print(f"✅ Input tensor shape: {image.shape}")
        
        # Inference
        input_tensor = torch.from_numpy(image).unsqueeze(0).to(device)
        
        print("🚀 Running inference...")
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                logits = model(input_tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                probs = torch.softmax(logits, dim=1)
                pred_mask = logits.argmax(1).cpu().numpy()[0]
                conf_score = probs.max(1)[0].mean().item() * 100
        
        current_prediction = pred_mask
        
        # DEBUG: Show prediction distribution
        unique_classes, class_counts = np.unique(pred_mask, return_counts=True)
        print(f"\n✅ Prediction complete! Confidence: {conf_score:.2f}%")
        print("📊 Predicted Class Distribution:")
        for cls, count in zip(unique_classes, class_counts):
            percentage = (count / pred_mask.size) * 100
            class_names = {0: "Background", 1: "Necrotic", 2: "Edema", 3: "Enhancing"}
            print(f"   Class {cls} ({class_names.get(cls, 'Unknown')}): {count:,} voxels ({percentage:.2f}%)")
        
        # Find best slices
        best_axial, best_sagittal, best_coronal = find_best_slices(pred_mask)
        print(f"\n🎯 Best slices - Axial: {best_axial}, Sagittal: {best_sagittal}, Coronal: {best_coronal}")
        
        # Calculate metrics
        tumor_voxels = np.sum(pred_mask > 0)
        necrotic_voxels = np.sum(pred_mask == 1)
        edema_voxels = np.sum(pred_mask == 2)
        enhancing_voxels = np.sum(pred_mask == 3)
        volume_ml = tumor_voxels * 0.001
        
        print(f"📊 Total Tumor: {tumor_voxels:,} voxels")
        print(f"   🔴 Necrotic: {necrotic_voxels:,}")
        print(f"   🟢 Edema: {edema_voxels:,}")
        print(f"   🔵 Enhancing: {enhancing_voxels:,}\n")
        
        # Classification
        if tumor_voxels < 100:
            diagnosis = "✅ No Significant Tumor Detected"
            color = "#00ff00"
            grade = "Healthy"
        elif enhancing_voxels > 500:
            diagnosis = "⚠️ HIGH-GRADE GLIOMA (Aggressive)"
            color = "#ff4444"
            grade = "Grade III/IV"
        else:
            diagnosis = "⚠️ LOW-GRADE GLIOMA"
            color = "#ffa500"
            grade = "Grade I/II"
        
        # Generate ALL THREE visualizations
        print("🎨 Generating visualizations for all three views...")
        
        axial_img = visualize_axial(best_axial)
        if axial_img is None:
            print("❌ ERROR: Axial visualization failed!")
        else:
            print(f"✅ Axial generated: {axial_img.size}")
        
        sagittal_img = visualize_sagittal(best_sagittal)
        if sagittal_img is None:
            print("❌ ERROR: Sagittal visualization failed!")
        else:
            print(f"✅ Sagittal generated: {sagittal_img.size}")
        
        coronal_img = visualize_coronal(best_coronal)
        if coronal_img is None:
            print("❌ ERROR: Coronal visualization failed!")
        else:
            print(f"✅ Coronal generated: {coronal_img.size}")
        
        # Report HTML
        report_html = f"""
        <div style="background: linear-gradient(135deg, #2b2b40 0%, #1a1a2e 100%); 
                    padding: 30px; border-radius: 15px; color: white; 
                    border-left: 5px solid {color}; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <h2 style="margin:0; color:{color}; font-size: 28px; font-weight: bold;">{diagnosis}</h2>
            <p style="color: #bbb; font-size: 16px; margin-top: 8px;">
                🏷️ {grade} | 🎯 Confidence: {conf_score:.1f}%
            </p>
            <hr style="border-color: #555; margin: 20px 0;">
            
            <h3 style="color: #4dd0e1; margin-bottom: 15px;">📊 Tumor Metrics</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;">
                    <p style="margin: 8px 0;"><b>📦 Total Volume:</b> {tumor_voxels:,} voxels</p>
                    <p style="margin: 8px 0;"><b>💉 Volume (mL):</b> {volume_ml:.2f} mL</p>
                    <p style="margin: 8px 0;"><b>📏 Percentage:</b> {(tumor_voxels/(128*128*128)*100):.2f}%</p>
                </div>
                <div style="background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px;">
                    <p style="margin: 8px 0;"><b>🔴 Necrotic Core:</b> {necrotic_voxels:,} voxels</p>
                    <p style="margin: 8px 0;"><b>🟢 Edema Region:</b> {edema_voxels:,} voxels</p>
                    <p style="margin: 8px 0;"><b>🔵 Enhancing Tumor:</b> {enhancing_voxels:,} voxels</p>
                </div>
            </div>
            
            <hr style="border-color: #555; margin: 20px 0;">
            
            <h3 style="color: #4dd0e1; margin-bottom: 10px;">🎚️ Best Viewing Slices</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                <div style="background: rgba(66,165,245,0.1); padding: 10px; border-radius: 8px; text-align: center;">
                    <p style="margin: 5px 0; font-size: 14px;"><b>Axial (Top-Down)</b></p>
                    <p style="margin: 5px 0; font-size: 18px; color: #42a5f5;">Slice {best_axial}/127</p>
                </div>
                <div style="background: rgba(102,187,106,0.1); padding: 10px; border-radius: 8px; text-align: center;">
                    <p style="margin: 5px 0; font-size: 14px;"><b>Sagittal (Side View)</b></p>
                    <p style="margin: 5px 0; font-size: 18px; color: #66bb6a;">Slice {best_sagittal}/127</p>
                </div>
                <div style="background: rgba(255,167,38,0.1); padding: 10px; border-radius: 8px; text-align: center;">
                    <p style="margin: 5px 0; font-size: 14px;"><b>Coronal (Front View)</b></p>
                    <p style="margin: 5px 0; font-size: 18px; color: #ffa726;">Slice {best_coronal}/127</p>
                </div>
            </div>
            
            <p style="margin-top: 20px; font-size: 14px; color: #aaa; text-align: center;">
                💡 Use the sliders in each view tab to explore all 128 slices
            </p>
        </div>
        """
        
        print("📤 Returning all 7 values to Gradio...\n")
        
        return (
            axial_img,
            sagittal_img,
            coronal_img,
            report_html,
            best_axial,
            best_sagittal,
            best_coronal
        )
        
    except Exception as e:
        import traceback
        error = traceback.format_exc()
        print(f"❌ Error: {error}")
        error_html = f"""
        <div style="background: #3d1f1f; padding: 20px; border-radius: 10px; color: white; border-left: 5px solid red;">
            <h2 style="margin:0; color:red;">❌ Error During Analysis</h2>
            <p style="color: #bbb; margin-top: 10px;"><b>{str(e)}</b></p>
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer;">Technical Details</summary>
                <pre style="background: #1a1a1a; padding: 10px; border-radius: 5px; overflow-x: auto; font-size: 11px;">{error}</pre>
            </details>
        </div>
        """
        return None, None, None, error_html, 64, 64, 64

# Load examples
print("\n📦 Loading example data...")
example_data = get_examples()

# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), css="""
    .gradio-container {max-width: 1600px !important}
    .tab-nav button {font-size: 16px; font-weight: bold;}
""") as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 20px;">
        <h1 style="font-size: 3.5em; margin: 0; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
            🧠 Brain Tumor Segmentation
        </h1>
        <p style="font-size: 1.3em; color: #f0f0f0; margin-top: 10px;">
            UNETR with Rotary Position Embeddings | Multi-View 3D Analysis
        </p>
    </div>
    """)
    
    with gr.Row():
        # LEFT COLUMN: Upload
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload MRI Scans")
            gr.Markdown("*NIfTI format (.nii / .nii.gz) | At least T1ce required*")
            
            t1_input = gr.File(label="T1 MRI")
            t1ce_input = gr.File(label="T1ce MRI (Contrast Enhanced) ⭐")
            t2_input = gr.File(label="T2 MRI")
            flair_input = gr.File(label="FLAIR MRI")
            
            with gr.Row():
                analyze_btn = gr.Button("🔍 Analyze Scan", variant="primary", size="lg")
                clear_btn = gr.ClearButton(
                    components=[t1_input, t1ce_input, t2_input, flair_input],
                    value="🗑️ Clear",
                    size="lg"
                )
            
            # Examples
            if example_data:
                gr.Markdown("---")
                gr.Markdown("### 📝 Try Example Data")
                gr.Examples(
                    examples=example_data,
                    inputs=[t1_input, t1ce_input, t2_input, flair_input],
                    label="Load Example Patient"
                )
        
        # RIGHT COLUMN: Results
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Results")
            diagnostic_report = gr.HTML()
            
            # Multi-view tabs
            with gr.Tabs() as tabs:
                with gr.TabItem("🔵 Axial View (Top-Down)"):
                    axial_output = gr.Image(type="pil", label="Axial Plane")
                    axial_slider = gr.Slider(0, 127, value=64, step=1, label="Axial Slice Navigator", interactive=True)
                
                with gr.TabItem("🟢 Sagittal View (Side)"):
                    sagittal_output = gr.Image(type="pil", label="Sagittal Plane")
                    sagittal_slider = gr.Slider(0, 127, value=64, step=1, label="Sagittal Slice Navigator", interactive=True)
                
                with gr.TabItem("🟠 Coronal View (Front)"):
                    coronal_output = gr.Image(type="pil", label="Coronal Plane")
                    coronal_slider = gr.Slider(0, 127, value=64, step=1, label="Coronal Slice Navigator", interactive=True)
    
    # Event handlers
    analyze_btn.click(
        fn=predict_and_analyze,
        inputs=[t1_input, t1ce_input, t2_input, flair_input],
        outputs=[axial_output, sagittal_output, coronal_output, diagnostic_report, 
                axial_slider, sagittal_slider, coronal_slider],
        show_progress="full"
    )
    
    # Use .release() instead of .change() to prevent auto-triggering
    axial_slider.release(fn=visualize_axial, inputs=[axial_slider], outputs=[axial_output])
    sagittal_slider.release(fn=visualize_sagittal, inputs=[sagittal_slider], outputs=[sagittal_output])
    coronal_slider.release(fn=visualize_coronal, inputs=[coronal_slider], outputs=[coronal_output])
    
    # Footer
    gr.Markdown("""
    ---
    ### 📌 How to Use:
    1. Upload MRI scans in NIfTI format (.nii or .nii.gz)
    2. At minimum, upload T1ce (contrast-enhanced) for analysis
    3. Click "Analyze Scan" and wait for processing (30-60 seconds on CPU)
    4. Explore different anatomical views using the tabs
    5. Use sliders to navigate through all 128 slices
    
    ### 🎨 Color Legend:
    - 🔴 **Red**: Necrotic/Non-enhancing tumor core
    - 🟢 **Green**: Peritumoral edema
    - 🔵 **Blue**: Enhancing tumor (aggressive component)
    
    ### 🎯 Anatomical Views:
    - **Axial**: Horizontal slices (top-down view)
    - **Sagittal**: Vertical slices from side (left-right)
    - **Coronal**: Vertical slices from front (front-back)
    
    ### ⚙️ Technical:
    - Architecture: UNETR with RoPE (~56M parameters)
    - Input: 4-channel MRI at 128³ resolution
    - Output: 4-class segmentation
    - Training: BraTS 2020 Dataset
    
    ### ⚠️ Disclaimer:
    Research tool only. NOT for clinical use. Consult medical professionals for diagnosis.
    
    ---
    <p style="text-align: center; color: #666;">Built with ❤️ using PyTorch 2.6 & Gradio 6.0</p>
    """)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Launching Brain Tumor Segmentation App...")
    print("="*60 + "\n")
    demo.launch()
