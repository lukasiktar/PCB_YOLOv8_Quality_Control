# PCB_YOLOv8_Quality_Control

This is a special Visual Studio project for object detection, extraction, inspection and OCR on printed circuit boards.

Project uses ONNX model of YOLOv8 custom neural network. The source provided by Ultralytics. Neural network is used to detect object of interest ( USB, Ethernet, HDMI ports, pins and their soldering patterns, text and other elements). OpenCV is used to load images and performing extraction and inspection. Inspection is performed using thresholding methods. For required detected elements OCR is performed. Results are shown in easy-to-use Qt GUI.

Project requirements: Visual studio 2019/2022 OpenCV 4.8.0 + CUDA compatible with OpenCV version CuDNN compatible with OpenCV and CUDA Qt 6.x.x configured in Visual studio Tesseract configured in Visual studio (used for OCR)
