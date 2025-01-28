# Disaster Response Pipeline Project

## Overview
This project is designed to assist disaster response teams by categorizing and organizing messages during emergency situations. It includes an ETL pipeline for processing data, a machine learning pipeline for message classification, and a Flask web application to provide real-time classification and visual insights.

The dataset, provided by [Figure Eight](https://www.figure-eight.com/) (formerly Appen), contains real-life disaster messages categorized into multiple needs such as medical assistance, water, shelter, and more. This project aims to streamline communication, enabling responders to allocate resources effectively.

---

## Dataset and Challenges

### Dataset Details
The dataset consists of two CSV files:
1. **Messages:** Contains the actual text messages related to disaster events.
2. **Categories:** Includes 36 binary columns indicating the type of need or action (e.g., `water`, `food`, `medical_help`).

### Data Cleaning
The ETL pipeline addresses common issues:
- Removing duplicates and null values.
- Splitting and converting categories into individual binary columns.
- Merging the messages and categories datasets for analysis.

### Challenges: Dataset Imbalance
Some categories (e.g., `water`, `medical_help`) have significantly fewer samples compared to others (e.g., `related`). This imbalance can lead to biased model predictions.

#### Model Training Considerations:
- **Emphasis on Precision or Recall:** For critical categories like `medical_help`, high recall is preferred to ensure no urgent need is missed.
- **Handling Imbalance:** Techniques such as oversampling, undersampling, or using class weights in the model were considered to improve performance.

---

## Project Components

1. **ETL Pipeline**
   - Extracts data from raw CSV files.
   - Cleans and transforms data into a structured SQLite database.

2. **Machine Learning Pipeline**
   - Builds a multi-output classifier using natural language processing (NLP) techniques like tokenization and lemmatization.
   - Outputs a trained model (`classifier.pkl`) for deployment.

3. **Flask Web Application**
   - Provides an intuitive interface for message classification.
   - Displays data insights with interactive visualizations.

---

## Instructions

### 1. Setting Up the Database and Model

Run the following commands in the project's root directory:

- **ETL Pipeline:**
  ```bash
  python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
  ```

- **Machine Learning Pipeline:**
  ```bash
  python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
  ```

### 2. Running the Web Application

Navigate to the app directory and start the Flask app:
```bash
python run.py
```

Access the app at: [http://0.0.0.0:3001](http://0.0.0.0:3001)

---

## Web App Features

1. **Message Classification**
   - Users can input disaster-related messages.
   - The app classifies messages into multiple relevant categories.

2. **Data Visualizations**
   - Real-time insights into message genres and category distributions.
   - Planned: Additional visualizations like category trends over time.

3. **Actionable Recommendations**
   - Based on classifications, emergency response organizations (e.g., NGOs) can prioritize actions like deploying water supplies or medical assistance.

4. **Responsive Design**
   - The app is user-friendly and accessible on mobile and desktop platforms.

---

## Deployment and Future Enhancements

- **Cloud Deployment:** The app can be deployed on AWS, Heroku, or Render for broader accessibility.
- **Improved Visualizations:** Adding time-series trends, category correlations, and prediction confidence levels.
- **Feedback Mechanism:** Allow users to correct misclassifications to improve model accuracy over time.
- **Customized UI:** Enhance the design for better user experience and branding.

---

## Dependencies

Install the required packages:
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `Flask`
- `pandas`
- `scikit-learn`
- `nltk`
- `SQLAlchemy`
- `plotly`

---

## File Structure

```
├── app
│   ├── templates
│   │   ├── go.html
│   │   └── master.html
│   └── run.py
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   └── process_data.py
├── models
│   ├── train_classifier.py
│   └── classifier.pkl
├── README.md
└── requirements.txt
```

---

## Licensing and Acknowledgements

This project was developed as part of the Udacity Data Science Nanodegree program. The dataset was provided by [Figure Eight](https://www.figure-eight.com/). Special thanks to the Udacity team for guidance and resources.

---

## Screenshots

*To be added*: Include screenshots of the web app interface, showing example classifications and data visualizations.
