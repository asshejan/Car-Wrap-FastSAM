# 🚗 Virtual Car Wrap - FastSAM

A powerful web application that uses FastSAM (Fast Segment Anything Model) to create virtual car wraps. Upload a car image and a sticker/decal, and see how it would look as a real car wrap!

## ✨ Features

- **AI-Powered Car Segmentation**: Uses FastSAM to automatically detect and segment cars from any background
- **Realistic Sticker Overlay**: Advanced alpha blending for realistic sticker placement
- **Web Interface**: User-friendly Streamlit interface
- **Download Results**: Save your virtual wraps as high-quality PNG images
- **GPU Support**: Automatic CUDA detection for faster processing
- **Multiple Format Support**: Works with JPG, JPEG, and PNG images

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
