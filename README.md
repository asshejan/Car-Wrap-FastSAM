# 🚗 Virtual Car Wrap - FastSAM

A professional Streamlit app for virtual car wrapping using AI-powered segmentation (FastSAM). Upload a car photo and a sticker/wrap design, and preview realistic wraps with advanced controls and effects.

## ✨ Features

- **AI Car Segmentation**: Uses FastSAM to segment the car from the background.
- **Automatic Body Masking**: Windows, tyres, and non-painted areas are automatically excluded from the wrap overlay for realism.
- **Sticker/Warp Overlay**: Upload a PNG wrap/sticker and see it mapped to the car's painted body only.
- **Advanced Controls**:
  - Horizontal & vertical wrap placement
  - Scale and rotation
  - Car color change
  - Paint texture (metallic, pearl, matte)
  - Professional backgrounds (studio, outdoor, garage)
  - Lighting and shadow enhancement
  - Photography filters (warm, cool, dramatic, vintage)
- **Input Validation**: Warns if you upload a logo or invalid car image.
- **Download Result**: Save your professional wrap preview as a PNG.

## 🛠️ Technology Stack

- **FastSAM**: Fast Segment Anything Model for car segmentation
- **Streamlit**: Web application framework
- **OpenCV**: Image processing and manipulation
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLO and SAM model support

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for faster processing)
- 4GB+ RAM recommended

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/asshejan/Car-Wrap-FastSAM.git
   cd Car-Wrap-FastSAM
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download FastSAM model** (if not already included)
   ```bash
   # The FastSAM-s.pt model should already be in the repository
   # If not, download it from the official FastSAM repository
   ```

## 🎯 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser**
   - Navigate to `http://localhost:8501`
   - The app will automatically open in your default browser

3. **Upload images**
   - **Car Image**: Upload a photo of a car (JPG, JPEG, PNG)
   - **Sticker/Decal**: Upload your sticker or decal image (PNG recommended for transparency)

4. **Generate wrap**
   - Click "Wrap it!" to process the images
   - Wait for the AI to segment the car and overlay the sticker
   - View the result and download if satisfied

## 📸 How It Works

1. **Car Segmentation**: FastSAM analyzes the car image and creates a precise mask of the vehicle
2. **Sticker Processing**: The sticker image is resized to fit the car's detected region
3. **Alpha Blending**: Advanced compositing techniques ensure realistic sticker placement
4. **Result Generation**: The final image shows how the wrap would look in reality

## 🎨 Tips for Best Results

- **Car Images**: Use clear, well-lit photos with the car as the main subject
- **Stickers**: PNG format with transparency works best
- **Resolution**: Higher resolution images produce better results
- **Background**: Simple backgrounds help with car detection

## 🔧 Configuration

You can customize the FastSAM model path in the sidebar:
- Default: `FastSAM-s.pt`
- Supports any compatible FastSAM checkpoint

## 📁 Project Structure

```
Car-Wrap-FastSAM/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── FastSAM-s.pt       # FastSAM model weights
├── README.md          # Project documentation
└── .gitignore         # Git ignore rules
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM) - Fast Segment Anything Model
- [Streamlit](https://streamlit.io/) - Web application framework
- [Ultralytics](https://ultralytics.com/) - YOLO and SAM implementations

## 📞 Support

If you encounter any issues or have questions, please open an issue on GitHub.

---

**Made with ❤️ using FastSAM and Streamlit**

## How the Masking Works

- The app uses FastSAM to segment the car.
- A refined mask is computed to focus on the painted body, excluding windows, tyres, and background using color and morphological filtering.
- The sticker/wrap image is automatically masked so it only appears on the car's painted panels—no wrap on glass or wheels!

## Best Practices for Sticker Images

- Use a transparent PNG for your wrap/sticker design.
- The design should be shaped or sized to fit the car's side view for best results.
- Avoid white backgrounds—use transparency where there is no wrap.

## Example

![screenshot](docs/example.png)

## Changelog

- **2024-07-05**: Improved realism—sticker overlay now automatically removes windows and tyres using the car body mask. Added vertical placement control.
- **2024-07-04**: Advanced wrap warping, blending, and pro controls.
- **2024-07-03**: Initial release.
