# TrafficSignsAI-Python

A full pipeline for building a custom traffic sign detection model using Wikipedia-sourced images and deploying it via ONNX for Android. Includes dataset creation, augmentation, training with YOLOv8, model conversion and quantization, and Android inference setup.

## 📁 Project Structure

```
TrafficSignsAI-Python/
│
├── wikipedia_dataset_updated.py       # Dataset scraper and preprocessor
├── data_analysis_attempt_wikipedia.ipynb # Data Analysis, YOLOv8 training and evaluation
├── data.yaml                          # YOLO class configuration
├── requirements.txt                   # Python dependencies
├── runs/                              # YOLO training results
└── wikipedia_dataset1/                # Final structured dataset (images/labels)
```

## 🚀 Features

- Scrapes road sign images from Wikipedia categories (China, Japan, Korea, etc.)
- Auto-generates bounding boxes and class labels
- Augments data with Albumentations: flips, brightness, rotation, etc.
- Balances class samples (aim: 50–150 per class)
- Trains YOLOv8 (Ultralytics) for object detection
- Converts and quantizes model to ONNX format
- Supports deployment to Android via ONNX Runtime Mobile (Kotlin)

## 🧩 Requirements

- Python ≥ 3.10
- PyTorch, Ultralytics YOLOv8
- Albumentations
- OpenCV, Requests, BeautifulSoup
- ONNX, onnxruntime, onnxruntime-tools
- Chrome + ChromeDriver (for scraping)