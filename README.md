# MaternaTrace AI

## Software-Based Maternal Health Risk Prediction Using Machine Learning

MaternaTrace AI is a software-based machine learning system that analyzes routinely collected maternal and pregnancy-related data to estimate birth-weight risk categories

The project explores how machine learning can support early identification of potentially concerning patterns using data that can be collected through routine healthcare interactions, without requiring additional hardware such as ultrasound devices, sensors, or wearables.

The system combines a machine learning research pipeline with an API backend and a web-based interface.

> **Important:** MaternaTrace AI is a research and educational prototype. It provides statistical predictions and is not a medical diagnostic system or a replacement for qualified healthcare professionals.

---

## Problem

Maternal and pregnancy-related health information is often collected across multiple clinical variables such as maternal age, BMI, blood pressure, hemoglobin level, prenatal visits, diabetes, hypertension, smoking status, and other factors.

Interpreting these variables together can be difficult when relying only on individual measurements.

MaternaTrace AI investigates whether machine learning can use these routinely collected variables to identify patterns associated with **birth-weight categories**, providing an additional data-driven signal that could support healthcare review.

The goal is not to replace clinical decision-making, but to demonstrate how a low-cost software system can transform structured maternal-health data into an interpretable machine learning prediction.

---

## Objective

The main objectives of MaternaTrace AI are to:

- Build a reproducible machine learning pipeline for maternal-health data.
- Predict the birth-weight category as **Normal** or **Low**.
- Compare multiple classification algorithms.
- Evaluate model performance using multiple statistical metrics.
- Apply explainability techniques to understand influential input features.
- Expose the trained model through a Flask API.
- Provide a web interface for interacting with the prediction system.
- Maintain a clear separation between machine learning prediction and clinical diagnosis.

---

## Technical Approach

The system follows a complete machine learning workflow:


Maternal Health Dataset
        ↓
Data Auditing
        ↓
Data Preprocessing
        ↓
Feature Engineering / Encoding
        ↓
Train-Test Split
        ↓
Stratified 5-Fold Cross-Validation
        ↓
Model Comparison
        ↓
Final Model Selection
        ↓
Model Evaluation
        ↓
Feature Importance Analysis
        ↓
Serialized Model
        ↓
Flask REST API
        ↓
Web Interface
