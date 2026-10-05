# NYC Yellow Taxi Trip Records — Big Data Analysis

A comprehensive machine learning and big data analysis of NYC Yellow Taxi trip records from January 2024. This project applies Apache Spark and Databricks to clean, analyse, and model large-scale taxi trip data (~2.8 million records) for revenue prediction and operational efficiency optimisation.

## Project Overview

This project demonstrates end-to-end big data analytics on real-world taxi trip data, covering data ingestion through predictive modelling. The analysis focuses on two key business problems:

1. **Revenue Prediction**: Predict total fare amount based on trip characteristics
2. **Operational Efficiency**: Estimate revenue per minute to identify high-efficiency trips

The project showcases practical applications of distributed computing, feature engineering, and ensemble machine learning techniques on datasets too large for traditional single-machine approaches.

## Dataset

- **Source**: [NYC Yellow Taxi Trip Records (January 2024)](https://www.kaggle.com/datasets/shayanshahid997/yellow-taxi-trip-record-of-january-2024)
- **Records**: ~2.8 million taxi trips
- **Time Period**: January 2024

### Key Features
- VendorID, pickup/dropoff datetime and location
- Passenger count, trip distance
- Fare, tip, tax, surcharge amounts
- Payment method and rate code

## Methodology

### 1. Data Ingestion & Cleaning
- Load data from Databricks-hosted CSV into Spark DataFrame
- Remove duplicates and handle null values in critical fields
- Filter outliers (zero distance, zero fare, unrealistic amounts)
- Convert categorical variables using indexing and one-hot encoding
- Create ML-ready feature pipeline for consistency

### 2. Exploratory Data Analysis (EDA)
- Analyse distributions of key metrics (fare, distance, tip amounts)
- Compare vendor performance and payment methods
- Geographic analysis of pickup/dropoff locations
- Temporal patterns (hour of day, weekday effects)
- Compute correlation matrix to identify relationships

### 3. Feature Engineering
Derived features include:
- `pickuphour` — Hour of day from pickup timestamp
- `tripdurationmin` — Trip duration in minutes
- `speedmph` — Estimated speed (distance/duration)
- `isweekend` — Binary indicator for weekend trips
- `revpermin` — Revenue per minute (efficiency metric)
- Location encodings from pickup/dropoff IDs

### 4. Machine Learning Models

#### Revenue Prediction (Target: `total_amount`)
- **Linear Regression**: Baseline model for interpretability
  - R² = 0.887, RMSE = $7.32, MAE = $3.53
- **Random Forest**: Non-linear alternative
  - R² = 0.885, RMSE = $7.38, MAE = $3.03

#### Operational Efficiency (Target: `rev_per_min`)
- **Random Forest**: Optimized for revenue-per-minute prediction
  - R² = 0.860, RMSE = 0.264, MAE = 0.183

### 5. Hyperparameter Tuning
- Manual grid search over key parameters
- Revenue models: Tested tree depth (5, 10, 15) and count (10, 50, 100)
- Efficiency model: Tuned forest size (10, 30, 50) and depth (5, 7, 10)
- Metrics: RMSE, Mean Absolute Error (MAE), R² score

## Results

### Final Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Revenue - Linear Regression | 7.32 | 3.53 | **0.887** |
| Revenue - Random Forest | 7.38 | 3.03 | 0.885 |
| Efficiency - Random Forest | 0.26 | 0.18 | **0.860** |

### Key Findings

- **Linear Regression** achieved the highest R² (88.7%) for revenue prediction, suggesting strong linear relationships with engineered features
- **Random Forest** produced lower MAE ($3.03 vs $3.53), indicating better median error for revenue predictions
- **Efficiency model** successfully captures revenue-per-minute patterns with R² of 0.86, useful for identifying high-value trips
- All models explain 85-89% of variance, demonstrating predictive power across conditions
- Feature engineering (derived speed, time-of-day, weekend indicator) significantly improved model accuracy

## Tools & Technologies

- **Apache Spark**: Distributed data processing and ML (SparkSQL, Spark DataFrames, SparkML)
- **Databricks Community Edition**: Cloud-based Spark execution and notebook interface
- **Python 3**: Data analysis and ML pipeline code
- **Pandas**: Data manipulation and results analysis
- **SparkML**: Regression models (Linear Regression, Random Forest)
- **MLflow** (implicit): Model evaluation and comparison

## Project Structure

**Repository Contents:**
```
├── NYC_Yellow_Taxi_Trip_Records.ipynb    # Main Databricks notebook
├── Report.pdf                            # Detailed project report
└── README.md                             # This file
```

**Dataset:**
- `nyc_tlc_yellow_2024_01.csv` — Download from [Kaggle](https://www.kaggle.com/datasets/shayanshahid997/yellow-taxi-trip-record-of-january-2024) (not included due to file size)

## Setup & Execution

### Prerequisites

- Databricks Community Edition account (free at [community.cloud.databricks.com](https://community.cloud.databricks.com))
- Dataset downloaded from [Kaggle](https://www.kaggle.com/datasets/shayanshahid997/yellow-taxi-trip-record-of-january-2024)
- No local installation required; runs entirely in the cloud

### Step 1: Download Dataset
1. Go to [NYC Yellow Taxi Trip Records (January 2024) on Kaggle](https://www.kaggle.com/datasets/shayanshahid997/yellow-taxi-trip-record-of-january-2024)
2. Sign in to your Kaggle account (or create one)
3. Download `nyc_tlc_yellow_2024_01.csv`

### Step 2: Create Databricks Account
1. Visit [https://community.cloud.databricks.com](https://community.cloud.databricks.com)
2. Sign up for a free account or log in

### Step 3: Upload Dataset
1. In Databricks sidebar: **+ New** → **Add or upload Data** → **Create or modify table** → **Upload File**
2. Select and upload the downloaded `nyc_tlc_yellow_2024_01.csv`
3. Note the DBFS path (typically `largedata.default.nyc_tlc_yellow_2024_01`)
4. Copy this path for use in the notebook

### Step 4: Import Notebook
1. In Databricks sidebar: **Workspace** → your username folder
2. Click dropdown → **Import**
3. Select and upload `NYC_Yellow_Taxi_Trip_Records.ipynb`

### Step 5: Run Notebook
1. Open the imported notebook
2. Update the data path in Section 1 if different from `largedata.default.nyc_tlc_yellow_2024_01`
3. Run all cells sequentially (or click **Run All** in toolbar)

#### Notebook Sections
| Section | Task |
|---------|------|
| 1 | Data Ingestion & Cleaning |
| 2 | Exploratory Data Analysis (EDA) |
| 3 | Feature Engineering |
| 4 | Machine Learning — Revenue Model |
| 5 | Machine Learning — Efficiency Model |
| 6 | Hyperparameter Tuning & Evaluation |

## Key Insights & Business Value

1. **Predictability**: Trip revenue is highly predictable (R² > 0.88) using basic trip features alone—useful for demand forecasting and pricing
2. **Feature Importance**: Distance, time-of-day, and passenger count dominate revenue predictions
3. **Efficiency Modeling**: The efficiency model identifies high-value trips (revenue per minute), enabling intelligent dispatch and trip acceptance
4. **Scalability**: Spark-based pipeline handles millions of records efficiently; easily extends to full-year or multi-year datasets
5. **Real-time Applications**: Models could be deployed for real-time fare estimates and operational recommendations

## Additional Resources

- **Report**: See `Report.pdf` for detailed findings, methodology, results, and discussion
- **Databricks Docs**: [Apache Spark on Databricks](https://docs.databricks.com/)
- **SparkML Docs**: [PySpark ML Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
