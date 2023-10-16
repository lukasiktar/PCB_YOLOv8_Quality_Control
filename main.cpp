/*
This is OpenCV project for detection, localization, inspection and OCR of PCBs (Printed Circuit Boards).
The project requires installation and importing some libraries:
* OpenCV with CUDA and cuDNN
* Tesseract with Leptonica

Developed by Luka Siktar
10.05.2023.
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include "MyWidget.h"
#include "colors.h"
#include "inference.h"
#include "Inspection.h"
#include "OCRcustom.h"


int main(int argc, char* argv[])
{
    //Initialization of Qt GUI application:
    QApplication app(argc, argv);
    MyWidget widget;


    //Creating an instance for performing inference (loading YOLOv8 neural network and file with its classes)
    bool runOnGPU = false;
    std::string projectBasePath = "Path\\to\\specified\\folder";
    Inference inf(projectBasePath + "\\source\\models\\YOLOv8_PCB.onnx", cv::Size(640, 640), projectBasePath + "\\source\\classes\\YOLOv8m_PCB_classes.txt", runOnGPU);

    //Additional modification for constant detection and bounding box colors
    Colors color(projectBasePath + "\\source\\classes\\YOLOv8m_PCB_classes.txt");
    std::map<std::string, cv::Scalar> dictionary = color.dictionary;

    //Inintialization of video stream
    std::string videoStreamURL = "http://192.168.8.101:8080/video";
    cv::VideoCapture cap(videoStreamURL);
    if (!cap.isOpened()) {
        std::cout << "Failed to connect to the IP Webcam stream." << std::endl;
        return -1;
    }

    //Segment of code performed after the "Capture" button is pressed
    cv::Mat capturedFrame;
    QObject::connect(&widget, &MyWidget::captureRequested, [&]() {
        cap.read(capturedFrame);
        //Check if captured image is sent and read properly
        if (!capturedFrame.empty()) {
            //Running the inference using YOLOv8
            std::vector<Detection> output = inf.runInference(capturedFrame);
            int detections = output.size();

            std::vector<cv::Mat> inspections;       //Stores all the inspecions images
            std::vector<cv::Mat> OCR_read_images;   //Stores all the images for OCR inspection
            std::vector<std::string> OCR_reads;     //Stores all reads for OCR_read_images
            std::vector<int> inspections_num;
            std::vector<std::string> inspections_name;

            //Loop through detections
            int a = 0;
            for (int i = 0; i < detections; ++i)
            {
                Detection detection = output[i];
                cv::Rect box = detection.box;                   //bounding box for detection
                Scalar color = dictionary[detection.className]; //color for bounding box

                Mat image1 = capturedFrame(box).clone();    //Extraction of object from captured image
                detection.detection_id = a++;                   //rewrite detection class_id-s to enable inspection of element with the same class_name and class_id          

                //Inspection
                if (detection.className == "40_pins" or detection.className == "6_pins" or detection.className == "Check_pattern_1" or detection.className == "Check_pattern_2" or detection.className == "Check_pattern_3" or detection.className == "Check_pattern_4" or detection.className == "CN7" or detection.className == "CN8" or detection.className == "CN9" or detection.className == "CN10") {
                    Inspection photo;
                    inspections.push_back(photo.inspect(image1, detection));
                    inspections_num.push_back(photo.boxes_number);
                    inspections_name.push_back(detection.className);
                }

                //OCR read
                if (detection.className == "ARDUINO" or detection.className == "UNO_white" or detection.className == "NVIDIA." or detection.className == "Arduino_UNO_model" or detection.className == "RaspberryPi_model" or detection.className == "STM32_model") {
                    OCRread OCRobject(image1);
                    OCR_reads.push_back(OCRobject.outputText);
                    if (image1.cols > 100 or image1.rows > 100) {
                        cv::resize(image1, image1, cv::Size(image1.cols / 2, image1.rows / 2));
                    }
                    OCR_read_images.push_back(image1);
                }
            }
            widget.showInspections(inspections, inspections_name, inspections_num);
            widget.showOCRdetections(OCR_read_images, OCR_reads);


            //Draw the bounding boxes and labels on captured image
            for (int i = 0; i < detections; ++i)
            {
                Detection detection = output[i];
                cv::Rect box = detection.box;
                cv::Scalar color = dictionary[detection.className];

                // Detection bounding box
                cv::rectangle(capturedFrame, box, color, 2);

                // Detection bounding box text
                //std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);  //Class name and confidence
                std::string classString = detection.className;  //Class name

                cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

                cv::rectangle(capturedFrame, textBox, color, cv::FILLED);
                cv::putText(capturedFrame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

            }
        }
        cv::resize(capturedFrame, capturedFrame, cv::Size(capturedFrame.cols / 3, capturedFrame.rows / 3));

        widget.caputureDisplayImage(capturedFrame);
        });
    //Segment of a code performed after "Exit" button is pressed
    QObject::connect(&widget, &MyWidget::exitRequested, [&]() { app.closeAllWindows();
    cap.release();
        });


    //Main infinite loop ( loop reads image and streams it to the GUI)
    while (true) {
        cv::Mat frame;
        cap.read(frame);
        //Check if the there is any frame to read
        if (frame.empty())
        {
            qDebug() << "Error: Unable to load image!";
            return -1;
        }

        cv::resize(frame, frame, cv::Size(frame.cols / 8, frame.rows / 8));
        //Streaming frame to the GUI
        widget.videoDisplayImage(frame);
        cv::waitKey(2);

        //Display GUI
        widget.show();
    }

    return app.exec();
}

//Additional function for transforming QImage format (used for Qt GUI) to Mat format (used for OpenCV)
cv::Mat QImageToCvMat(const QImage& image)
{
    cv::Mat mat;
    switch (image.format()) {
    case QImage::Format_RGB888:
        mat = cv::Mat(image.height(), image.width(), CV_8UC3, const_cast<uchar*>(image.bits()), image.bytesPerLine());
        cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB); // Optional: Convert BGR to RGB format
        break;
    default:
        mat = cv::Mat();
        break;
    }
    return mat;
}

//Additional function for transforming Mat format (used fot OpenCV) to QImage format (used for GUI)
QImage CvMatToQImage(const cv::Mat& mat)
{
    QImage image;
    switch (mat.type()) {
    case CV_8UC3:
        image = QImage(mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);
        break;
    default:
        image = QImage();
        break;
    }
    return image;
}