# 🖤🎨 BW COLORIZER

A GUI application for colorizing **black & white** photos into **full-color** images using the SIGGRAPH AI model (Caffe + OpenCV DNN).

## 📌 Key Features
- Automatic colorization from B/W to full color  
- Side-by-side Before / After split view  
- Drag image with left mouse button  
- Dark mode interface  
- Save colorized results  

## 📥 Required Model Files
Place all files inside the `models/` folder.

### 1️⃣ colorization_release_v2.caffemodel  
https://huggingface.co/spaces/BilalSardar/Black-N-White-To-Color/blob/main/colorization_release_v2.caffemodel

## 📂 Folder Structure
BW-COLORIZER/  
│  
├── main.py  
├── models/  
│   ├── colorization_deploy_v2.prototxt  
│   ├── colorization_release_v2.caffemodel  
│   └── pts_in_hull.npy  
│  
└── images

## 🔧 Make sure you have Python and required libraries installed.
To install dependencies:
```bash
pip install pillow opencv-python numpy
