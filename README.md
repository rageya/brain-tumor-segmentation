# 🧠 Brain Tumor Segmentation with UNETR

AI-powered brain tumor segmentation using **UNETR** (Transformers for 3D Medical Imaging) enhanced with **Rotary Position Embeddings (RoPE)** for superior 3D spatial awareness.

[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-blue)](https://huggingface.co/spaces/Rageya/brain-tumor-segmentation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Features

- **Multi-modal MRI support** - Accepts T1, T1ce, T2, and FLAIR sequences
- **Automatic tumor classification** - Distinguishes between low-grade and high-grade gliomas
- **3D visualization** - Color-coded tumor regions for clear interpretation
- **Confidence scoring** - Provides prediction confidence and volume metrics
- **Real-time inference** - GPU-accelerated processing for fast results
- **Clinical-grade output** - Generates comprehensive diagnostic reports

## 📊 Model Architecture

- **Base Model:** UNETR (UNet + Vision Transformer)
- **Enhancement:** Rotary Position Embeddings (RoPE) for improved 3D spatial awareness
- **Parameters:** ~56 million
- **Training Dataset:** BraTS 2020 (Multimodal Brain Tumor Segmentation Challenge)
- **Performance:** Dice Score ~0.87 on validation set
- **Input:** 4-channel 3D MRI volumes (155 slices, 240×240 pixels)
- **Output:** 3-class segmentation masks

## 🎯 How to Use

### Online Demo
Visit the [Hugging Face Space](https://huggingface.co/spaces/Rageya/brain-tumor-segmentation) for instant testing.

### Local Installation
```bash
# Clone the repository
git clone https://github.com/Rageya/brain-tumor-segmentation.git
cd brain-tumor-segmentation

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

### Input Requirements

- Upload MRI scans in NIfTI format (.nii or .nii.gz)
- Minimum requirement: T1ce (contrast-enhanced) scan
- For best results: Upload all four modalities (T1, T1ce, T2, FLAIR)
- Click "Analyze Scan" to generate segmentation

## 🎨 Segmentation Classes

The model identifies three distinct tumor regions:

| Class | Region | Color | Clinical Significance |
|-------|--------|-------|----------------------|
| Class 1 | Necrotic/Non-enhancing tumor core | 🔴 Red | Central dead tissue |
| Class 2 | Peritumoral edema | 🟢 Green | Swelling around tumor |
| Class 3 | Enhancing tumor | 🔵 Blue | Aggressive/active tumor |

## 📦 Technical Stack

- PyTorch 2.6.0
- Gradio 6.0.1
- NiBabel 5.3.0
- NumPy 1.26+
- SciPy 1.13+
- Matplotlib 3.8+

## 🏗️ Project Structure
```
brain-tumor-segmentation/
├── app.py                 # Gradio interface
├── model.py              # UNETR architecture with RoPE
├── preprocessing.py      # Data preprocessing utilities
├── requirements.txt      # Python dependencies
├── checkpoint.pth        # Trained model weights
└── README.md            # This file
```

## 📈 Training Details

- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-5)
- **Loss Function:** Dice Loss + Cross-Entropy
- **Batch Size:** 2 (due to 3D volume memory constraints)
- **Epochs:** 300
- **Augmentation:** Random flips, rotations, intensity shifts
- **Hardware:** NVIDIA GPU with 16GB+ VRAM

## 📚 Citations

### Model Architecture

This project implements the UNETR architecture. If you use this work, please cite:
```bibtex
@inproceedings{hatamizadeh2022unetr,
  title={UNETR: Transformers for 3D Medical Image Segmentation},
  author={Hatamizadeh, Ali and Tang, Yucheng and Nath, Vishwesh and Yang, Dong and Myronenko, Andriy and Landman, Bennett and Roth, Holger R and Xu, Daguang},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={574--584},
  year={2022},
  organization={IEEE}
}
```

### Dataset

This model was trained on the BraTS 2020 dataset. Please cite all three papers as required:
```bibtex
@article{menze2015multimodal,
  title={The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)},
  author={Menze, Bjoern H and Jakab, Andras and Bauer, Stefan and Kalpathy-Cramer, Jayashree and Farahani, Keyvan and Kirby, Justin and Freymann, John and others},
  journal={IEEE Transactions on Medical Imaging},
  volume={34},
  number={10},
  pages={1993--2024},
  year={2015},
  publisher={IEEE},
  doi={10.1109/TMI.2014.2377694}
}

@article{bakas2017advancing,
  title={Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features},
  author={Bakas, Spyridon and Akbari, Hamed and Sotiras, Aristeidis and Bilello, Michel and Rozycki, Martin and Kirby, Justin S and Freymann, John and Farahani, Keyvan and Davatzikos, Christos},
  journal={Scientific Data},
  volume={4},
  number={1},
  pages={170117},
  year={2017},
  publisher={Nature Publishing Group},
  doi={10.1038/sdata.2017.117}
}

@article{bakas2018identifying,
  title={Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge},
  author={Bakas, Spyridon and Reyes, Mauricio and Jakab, Andras and Bauer, Stefan and Rempfler, Markus and Crimi, Alessandro and Shinohara, Russell T and others},
  journal={arXiv preprint arXiv:1811.02629},
  year={2018}
}
```

## ⚠️ Medical Disclaimer

**IMPORTANT:** This is a research and educational tool designed for academic purposes only. It is **NOT** approved for clinical use and should **NOT** be used for medical diagnosis or treatment decisions.

- Results are for research demonstration only
- Always consult qualified medical professionals for diagnosis
- No warranty or guarantee of accuracy is provided
- Not FDA approved or clinically validated

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- UNETR architecture by Hatamizadeh et al. (NVIDIA & Vanderbilt University)
- BraTS Challenge organizers for the high-quality annotated dataset
- Medical imaging community for advancing AI in healthcare
- Hugging Face for hosting the demo space

## 📧 Contact

For questions or collaborations, please open an issue or reach out through GitHub.

---

Made with ❤️ for medical AI research | Star ⭐ this repo if you find it helpful!
