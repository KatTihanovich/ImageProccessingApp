# ImageProcessingApp

A graphical application for basic image processing with an interactive interface built using Tkinter.  
It allows users to apply various filters, transformations, and visualize the results.

This project is part of a **Computer Vision** course and demonstrates practical image processing techniques using OpenCV.

---

## Available Processing Methods

- Grayscale conversion (average RGB and HSV V channel)
- Binarization (fixed threshold and Otsu’s method)
- Histogram normalization and stretching
- Histogram equalization
- Blurring and sharpening (Gaussian Blur, Laplacian)
- Edge detection (Sobel)
- Image translation (horizontal and vertical shifting)
- Image rotation by an arbitrary angle

---

## Requirements

| Library / Language | Required Version |
|-------------------|-----------------|
| Python            | 3.10            |
| OpenCV (cv2)      | 4.5.0           |
| NumPy             | 1.21.0          |
| Pillow (PIL)      | 9.0.0           |

---

## Installation & Launch

1. Clone the repository:
*https://github.com/KatTihanovich/ImageProccessingApp.git*

2. Install the required libraries:
*pip install opencv-python numpy pillow*

3. Run the application:
- From your IDE using the **Run** button, or
- From the terminal:
  ```
  python image_processor_app.py
  ```
4. Alternatively, you can run the precompiled executable:
- `image_processing_app.exe` (double-click to launch)

---

## Usage

1. Load an image using the **Load Image** button
2. Select the desired image processing method
3. Click **Apply**
4. Click **Show All Steps** to visualize all processing stages
5. Adjust parameters if needed using the input fields
6. Save results using one of the save options:
- Save original
- Save grayscale
- Save processed

---

## Notes

- Some methods require parameter tuning for optimal results
- Best results are achieved with good-quality input images
