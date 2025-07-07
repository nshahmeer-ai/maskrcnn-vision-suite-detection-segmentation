# maskrcnn-vision-suite-detection-segmentation
A complete Mask R-CNN pipeline for object detection and instance segmentation on the COCO dataset, with visualization and mAP evaluation using PyTorch.

# 🧠 Mask R-CNN Object Detection and Instance Segmentation with mAP Evaluation

Welcome to the Mask R-CNN Vision Suite! This project demonstrates object detection and instance segmentation using the pretrained Mask R-CNN (ResNet-50 FPN) model on the COCO val2017 dataset. The system visualizes predictions, applies segmentation masks, and evaluates performance using mAP (mean Average Precision) via the COCO evaluation API.

---

## 📌 Overview

The Mask R-CNN Vision Suite project aims to showcase a complete end-to-end pipeline for object detection and instance segmentation using a pretrained model. It loads the COCO validation dataset, performs inference, visualizes results with masks and bounding boxes, and computes evaluation metrics using pycocotools.

---

## 🧱 Project Structure

The project is organized into clear, functional components for a smooth workflow:

- Notebook: A Jupyter Notebook that guides you through loading data, running inference, and evaluating results.
- Setup: Instructions for installing dependencies and downloading the COCO dataset.
- Model: Utilizes PyTorch's pretrained maskrcnn_resnet50_fpn from torchvision.models.detection.
- Evaluation: Uses COCOeval from pycocotools for calculating mAP scores.
- Visualization: Displays bounding boxes and masks over real COCO images using OpenCV and Matplotlib.

---

## 🌐 Key Features

- Object Detection with bounding boxes  
- Instance Segmentation using mask overlays  
- mAP Evaluation using COCO API (pycocotools)  
- Pretrained Mask R-CNN (ResNet-50 FPN backbone)  
- Visualization with OpenCV + Matplotlib  
- Auto-download COCO annotations & images  

---

## ⚙️ Setup

Follow these steps to get started:

### 1. Clone the Repository

```bash
git clone "https://github.com/nshahmeer-ai/maskrcnn-vision-suite-detection-segmentation"
cd maskrcnn-object-detection-segmentation
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv maskrcnn-env
source maskrcnn-env/bin/activate  # for Linux/macOS
maskrcnn-env\Scripts\activate     # for Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Launch Notebook

```bash
jupyter notebook
```

Open MaskRCNN_COCO_Inference.ipynb and follow the cells step by step.

---

## 📂 Dataset Download

The notebook will automatically download:

- COCO 2017 Annotations (instances_val2017.json)  
- COCO Validation Images (val2017)  

If already present, the download is skipped.

---

## 📊 mAP Evaluation

After inference, the results are saved in coco_results.json.  
Then COCOeval from pycocotools is used to calculate:

- AP@[IoU=.50:.95] (mAP)  
- AP@IoU=0.5  
- AR (Average Recall)  

These are standard metrics used for benchmarking models on COCO.

---

## 📸 Visualization Sample

Bounding boxes and masks are drawn over original images:

- Green rectangles: Detected objects  
- Red masks: Segmented regions  
- White text: Class IDs  

Matplotlib is used to render the final visualization.

---

## 🧠 Technologies Used

- PyTorch and TorchVision  
- Mask R-CNN (ResNet-50 FPN)  
- pycocotools (COCO evaluation API)  
- OpenCV, Matplotlib, Pillow, NumPy  

---

## 🧪 Usage

To test your own image or a different COCO image, change the image ID in the notebook:

```python
img_ids = coco.getImgIds()
img_info = coco.loadImgs(img_ids[0])[0]  # change index for different image
```

---


Contributing

Star this repo
Fork and PR improvements
Report issues or bugs
Add new model integrations or features
Open to:

Remote roles (ML/DL Engineer, CV Specialist)
Research internships or collaborations
Open-source AI/ML contributions

About the Author Shahmeer Nawaz Master's Student in Artificial Intelligence

Email: shahmeernawazai@gmail.com GitHub: https://github.com/nshahmeer-ai LinkedIn: https://www.linkedin.com/in/shahmeernawazai


