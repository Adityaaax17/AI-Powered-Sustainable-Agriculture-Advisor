# 🌱 AI-Powered Sustainable Agriculture Advisor

An AI-based decision-support prototype that analyzes agricultural data and provides insights to support productive and sustainable farming.

## 📌 Project Overview

Agriculture depends on several environmental and farming factors such as rainfall, temperature, humidity, soil conditions, crop type, season, farm area, and irrigation method.

Farmers may find it difficult to evaluate these factors together when making decisions related to crop productivity and resource usage.

The **AI-Powered Sustainable Agriculture Advisor** uses historical agricultural data and a Machine Learning model to analyze farming conditions, estimate crop yield, evaluate sustainability indicators, and provide data-driven farming insights.

The system is designed as a decision-support tool and does not replace farmers or agricultural experts.

## 🎯 Problem Statement

Farmers need to make decisions under changing environmental and resource conditions. Traditional decision-making may not fully utilize historical agricultural data to understand relationships between farming conditions, crop yield, and resource efficiency.

This project aims to provide an accessible AI-based system that analyzes historical agricultural records and uses Machine Learning to generate crop-yield predictions and sustainability-related insights.

## 🌍 Sustainable Development Goal

### SDG 2 — Zero Hunger

The project is aligned with **United Nations Sustainable Development Goal 2: Zero Hunger**.

The system supports this goal by using AI and agricultural data to help improve understanding of crop productivity and encourage more efficient and sustainable farming practices.

## 🤖 AI & Machine Learning

The project uses **Machine Learning**, particularly a:

### Random Forest Regression Model

The model is trained using historical agricultural records and is used to estimate crop yield based on selected agricultural and environmental conditions.

The prototype displays model performance using:

- R² Score
- Mean Absolute Error (MAE)
- Predicted Yield
- Historical Yield

The model shown in the prototype achieved an R² score of approximately **0.774** on the evaluated data.

## 🌱 Key Features

### 1. Farm Information Input

Users can provide:

- Crop
- Season
- State
- Irrigation method
- Rainfall
- Average temperature
- Humidity
- Farm area

### 2. AI Agricultural Analysis

The system provides:

- Predicted crop yield
- Historical yield
- Average profit
- Average water usage
- Water-efficiency information

### 3. Machine Learning Model

The application uses a Random Forest model to analyze historical agricultural records and estimate crop yield.

### 4. Sustainability Analysis

The system provides a sustainability indicator and status to help users understand resource-efficiency considerations.

### 5. AI-Generated Farming Insights

The application generates data-driven insights based on:

- Predicted yield
- Historical agricultural records
- Rainfall conditions
- Water-use efficiency

### 6. Agricultural Data Visualization

The dashboard includes visualizations such as:

- Average yield by crop
- Average yield by season
- Yield by irrigation method
- Water efficiency by irrigation method

### 7. Historical Records

Relevant historical agricultural records are displayed to provide context for the analysis.

### 8. Responsible AI

The prototype considers:

- **Fairness:** Historical data may not represent every farming region or crop equally.
- **Transparency:** Predictions are based on agricultural features used by the model.
- **Privacy:** The prototype avoids collecting unnecessary personal or sensitive farmer information.
- **Human Oversight:** The system provides decision support and does not replace farmers or agricultural experts.

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Random Forest Machine Learning
- CSV-based agricultural dataset

## 📊 Dataset

The project uses a historical agricultural performance dataset containing agricultural and environmental variables such as:

- Crop
- Season
- State
- Farm area
- Rainfall
- Average temperature
- Humidity
- Sunlight hours
- Soil pH
- Soil moisture
- Nitrogen
- Phosphorus
- Potassium
- Irrigation method
- Crop yield
- Profit
- Water usage

## 🖥️ Application Interface

The application provides an interactive Streamlit dashboard where users can enter farm conditions and analyze agricultural performance.

### Main Dashboard

The dashboard displays farm information, predicted yield, historical yield, profit, water usage, and water efficiency.

### Sustainability Analysis

The system provides a sustainability indicator and explains relevant farming insights.

### Data Visualization

Charts allow users to compare crop yield, seasons, irrigation methods, and water efficiency.

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Adityaaax17/AI-Powered-Sustainable-Agriculture-Advisor.git
