Parking Vacancy Detection System — MLOps Project

This project is an end-to-end Parking Vacancy Detection System built using Convolutional Neural Networks (CNNs) and a full MLOps pipeline for training, deployment, monitoring, and continuous improvement. The goal is to automatically detect whether a parking slot is occupied or vacant using camera images, enabling smarter traffic management and real-time parking assistance.

📌 Overview

This project implements an end-to-end Parking Vacancy Detection System using Convolutional Neural Networks (CNNs) combined with a complete MLOps workflow.
The system classifies each parking slot as “Free” or “Busy” using image data, and automatically handles training, versioning, deployment, monitoring, and retraining.

Built with a production-first mindset, the pipeline ensures the model remains reliable, scalable, and continuously improving.

🚀 Key Features

CNN-based parking spot classification

Automated data ingestion & preprocessing

MLflow experiment tracking & model registry

DVC-based dataset versioning

Docker containerization for reliable deployment

CI/CD using GitHub Actions

Deployment on AWS EC2 with AWS ECR

CloudWatch logging & monitoring

Data drift detection + automated retraining triggers

🧠 Model

TensorFlow/Keras CNN

Transfer learning for improved accuracy

Binary classification (free vs busy)

Monitored using softmax confidence scores

Versioned and stored in MLflow Model Registry

🛠 Tech Stack

ML: TensorFlow, NumPy, Pandas
MLOps: MLflow, DVC, GitHub Actions
Deployment: Docker, FastAPI/Flask, AWS EC2, AWS ECR
Monitoring: AWS CloudWatch
Version Control: Git + GitHub

📦 Pipeline Stages

Data ingestion & preprocessing

Model training + evaluation

Logging + versioning (MLflow)

Docker build

Push to ECR

Automatic deployment to EC2

Monitoring, drift detection, retraining

📊 Results

High accuracy on CNRPark+EXT dataset

Stable precision/recall

Low-latency inference via REST API

🔮 Future Work

Edge-optimized models (MobileNet/EfficientNet)

YOLO-based full-slot detection

Autoretraining with real-time feedback

Web dashboard for live monitoring
