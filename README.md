# 🌊 Caustics Water Surface Simulator

> A **physically-based water caustics simulator** designed for **real-time rendering** using Python, featuring advanced optics, chromatic dispersion, and interactive detection modes.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Accelerated-orange.svg)](https://numpy.org/)
[![Numba](https://img.shields.io/badge/Numba-JIT-green.svg)](https://numba.pydata.org/)
[![OpenCL](https://img.shields.io/badge/OpenCL-Optional-purple.svg)](https://www.khronos.org/opencl/)

![Water Caustics Animation](https://img.shields.io/badge/Real--Time-60+_FPS-brightgreen)

---

## ✨ Features

### 🌟 **Physically Accurate Optics**
- **Chromatic Dispersion**: Wavelength-dependent refraction (RGB channels: 650nm, 550nm, 450nm)
- **Cauchy's Equation**: Realistic index of refraction calculations
- **Beer-Lambert Law**: Wavelength-specific light absorption in water
- **Snell's Law**: Accurate ray tracing and refraction

### 🎨 **Advanced Rendering**
- **Multi-Light System**: Dual light sources with adjustable angles and intensities
- **Post-Processing Pipeline**:
  - ✓ ACES Tonemapping
  - ✓ Bloom effects with adjustable threshold
  - ✓ Subsurface Scattering (SSS)
  - ✓ Vignette
  - ✓ Exposure control
- **4K Export**: Render stunning 4096×4096 images with 4× supersampling
- **Video Export**: Generate 30 FPS MP4 animations

### 💧 **Realistic Water Simulation**
- **Multi-Octave Simplex Noise**: Natural wave patterns
- **Gerstner Waves**: Physically-based ocean wave simulation
- **Interactive Ripples**: Click to create propagating waves
- **Micro-Detail**: Fine surface perturbations for realism

### 🔍 **Interactive Detection System**
- **Coin Simulation**: Drop virtual coins and observe refraction effects
- **Object Detection**: Click apparent positions to calculate real positions
- **Snell's Law Correction**: Reverse ray-tracing to find true object locations
- **Visual Feedback**: Real-time visualization of real vs apparent positions

### ⚡ **Performance Optimizations**
- **Numba JIT Compilation**: Parallel CPU execution
- **OpenCL Support**: Optional GPU acceleration
- **Multi-threaded Rendering**: Responsive UI during computation
- **Real-time Preview**: 30-60 FPS simulation at 512×512

---

## 🎮 Presets

Choose from stunning pre-configured scenarios:

| Preset | Description |
|--------|-------------|
| **🌅 Calm Pool** | Gentle waves, soft caustics, perfect for serene scenes |
| **⛈️ Stormy Ocean** | Turbulent water, intense caustics, dramatic lighting |
| **🍮 Jell-O** | Viscous medium simulation, extreme SSS |
| **🛢️ Oil Slick** | Thin film interference-like effects, subtle waves |

---

## 🚀 Quick Start

### Prerequisites
```bash
pip install numpy numba opencv-python customtkinter pillow
```

**Optional GPU Acceleration:**
```bash
pip install pyopencl
```

### Run the Simulator
```bash
python caustics_pro.py
```

---

## 🎛️ Controls Overview

### **Waves Tab**
- Adjust wave scale, height, and micro-details
- Control Gerstner wave parameters (amplitude, wavelength, speed)

### **Lights Tab**
- Configure dual light sources
- Adjust angles (X/Y) and intensities
- Simulate sunrise, overhead sun, or dual lighting scenarios

### **Render Tab**
- Set water depth (affects absorption)
- Adjust exposure (EV stops)
- Pause/resume simulation

### **Post-FX Tab**
- Bloom threshold and intensity
- Vignette strength
- Subsurface scattering (SSS) intensity and radius

### **Detect Tab**
- Drop coins to test refraction detection
- Enable detection mode to find real positions
- Visualize the difference between apparent and actual locations

---

## 🧪 How It Works

### Ray Tracing Pipeline
1. **Surface Generation**: Multi-octave Simplex noise + Gerstner waves + interactive ripples
2. **Normal Calculation**: Finite difference method for surface gradients
3. **Refraction**: Snell's law applied per RGB wavelength (chromatic dispersion)
4. **Absorption**: Beer-Lambert law for underwater light attenuation
5. **Caustic Accumulation**: Bilinear splatting of refracted rays onto floor
6. **Post-Processing**: Bloom → Tonemapping → Vignette → Color grading

### Physics Constants
```python
WAVELENGTH_R = 650.0 nm  # Red light
WAVELENGTH_G = 550.0 nm  # Green light
WAVELENGTH_B = 450.0 nm  # Blue light

CAUCHY_A = 1.3247       # Cauchy coefficient A
CAUCHY_B = 3300.0       # Cauchy coefficient B

MU_R = 0.20  # Red absorption coefficient
MU_G = 0.05  # Green absorption coefficient
MU_B = 0.01  # Blue absorption coefficient
```

---

## 📊 Performance

| Resolution | FPS (CPU) | FPS (GPU)* |
|-----------|-----------|-----------|
| 256×256   | 45-60     | 80-120    |
| 512×512   | 15-25     | 40-60     |
| 1024×1024 | 3-5       | 12-20     |

*GPU acceleration requires PyOpenCL

---

## 🎨 Export Options

### 4K Still Image
- Click **🎨 4K** button
- Renders at 4096×4096 with 4× supersampling
- Saves as PNG with timestamp

### Video Animation
- Click **🎬 Video** button
- Generates 150-frame looping animation
- Exports as MP4 (30 FPS, 512×512)

---

## 🧬 Technical Highlights

- **Numba JIT Acceleration**: Critical functions compiled to machine code
- **Parallel Ray Tracing**: `@jit(parallel=True)` for multi-core utilization
- **Bilinear Interpolation**: Smooth caustic splatting
- **Wavelength Splitting**: Per-channel IOR for chromatic aberration
- **ACES Tonemapping**: Film-like HDR → LDR mapping
- **Inverse Refraction**: Novel algorithm to find real object positions from apparent ones

---

## 🔬 Use Cases

- **Computer Graphics Education**: Learn ray tracing, refraction, and light transport
- **Scientific Visualization**: Demonstrate Snell's law and optical phenomena
- **Game Development**: Generate caustic textures and reference materials
- **Physics Research**: Simulate underwater optics and light behavior
- **Art & Design**: Create stunning water caustic patterns for VFX

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Numba Team**: For incredible JIT compilation
- **Simplex Noise**: Ken Perlin's noise algorithm
- **Gerstner Waves**: Jerry Tessendorf's ocean simulation work
- **ACES**: Academy Color Encoding System for tonemapping

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs via Issues
- Suggest new features
- Submit pull requests
- Share your rendered caustics!

---

## 📫 Contact

Created by [@YokaMakuto](https://github.com/YokaMakuto)

**Enjoy creating beautiful water caustics! 🌊✨**