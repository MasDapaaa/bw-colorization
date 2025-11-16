# 🖤🎨 BW COLORIZER
![result-1](https://github.com/user-attachments/assets/10fa2827-219f-4cad-9667-961cf9c11435)
![result-2](https://github.com/user-attachments/assets/3604f70c-78ce-4386-ad34-c874d08488c8)
![result-3](https://github.com/user-attachments/assets/9518dce8-ddbc-44b3-a785-3b1ae00034b7)

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
