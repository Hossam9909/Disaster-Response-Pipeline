# Disaster Response Pipeline Project

This project is part of the Udacity Data Scientist Nanodegree. It uses a machine learning pipeline to categorize real disaster response messages into multiple categories. The web app allows users to input a message and get predicted categories to help guide emergency aid responses.

## Table of Contents

* [Project Motivation](#project-motivation)
* [Dataset and Challenges](#dataset-and-challenges)
* [Project Components](#project-components)
* [ETL Pipeline](#etl-pipeline)
* [Machine Learning Pipeline](#machine-learning-pipeline)
* [Flask Web Application](#flask-web-application)
* [Visualizations](#visualizations)
* [Model Performance](#model-performance)
* [Imbalanced Dataset Discussion](#imbalanced-dataset-discussion)
* [Recommendations System](#recommendations-system)
* [Instructions](#instructions)
* [Web App Features](#web-app-features)
* [Deployment](#deployment)
* [Installation](#installation)
* [File Structure](#file-structure)
* [Screenshots](#screenshots)
* [License and Acknowledgements](#license-and-acknowledgements)

---

## Project Motivation

During a natural disaster or crisis, thousands of messages are received via social media and other platforms. Emergency teams need an automated way to classify and prioritize these messages. This project builds a classifier that can assign categories such as "medical help," "water," and "shelter" to aid in disaster response.

The goal is to streamline communication during emergencies, enabling responders to allocate resources effectively and ensure critical needs are addressed promptly.

---

## Dataset and Challenges

### Dataset Details
The dataset is provided by [Figure Eight (CrowdFlower)](https://appen.com/) and consists of two CSV files:

1. **Messages (`disaster_messages.csv`):** Contains real-life disaster-related messages with metadata
2. **Categories (`disaster_categories.csv`):** Includes 36 binary columns indicating the type of need or action

**Dataset Statistics:**
- 26,000+ disaster-related messages
- 36 different categories for classification
- Multiple languages with English translations
- Real messages from various disaster events

### Example Data Structure:

| id | message | original | genre | water | medical_help | food | shelter | ... |
|----|---------|----------|-------|-------|--------------|------|---------|-----|
| 1 | "There is flooding in my house." | "Il y a des inondations dans ma maison." | direct | 1 | 0 | 0 | 1 | ... |
| 2 | "We need medical supplies urgently." | - | social | 0 | 1 | 0 | 0 | ... |

### Data Cleaning Challenges
The ETL pipeline addresses common data issues:
- Removing duplicates and handling null values
- Converting multi-label categories from string format to binary columns
- Handling edge cases where category values exceed binary format
- Merging datasets while preserving data integrity

### Dataset Imbalance Challenge
Some categories (e.g., `water`, `medical_help`) have significantly fewer samples compared to others (e.g., `related`, `aid_related`). This imbalance can lead to biased model predictions.

**Implications:**
- Model may achieve high accuracy but poor recall for rare categories
- For disaster scenarios, high recall is critical: missing a request for help is worse than false alarms
- Requires specialized techniques to handle minority class predictions

**Solutions Implemented:**
- `class_weight='balanced'` in RandomForestClassifier
- Comprehensive evaluation metrics for each category
- Focus on F1-score and recall for critical categories

---

## Project Components

This project consists of three main components designed to work together seamlessly:

### 1. ETL Pipeline (`process_data.py`)
- Extracts data from raw CSV files
- Transforms and cleans data with advanced preprocessing
- Loads processed data into SQLite database
- Includes comprehensive logging and error handling

### 2. Machine Learning Pipeline (`train_classifier.py`)
- Builds multi-output classifier using NLP techniques
- Implements advanced text processing (tokenization, lemmatization)
- Performs hyperparameter tuning with GridSearchCV
- Generates comprehensive evaluation reports and visualizations

### 3. Flask Web Application (`run.py`)
- Provides intuitive interface for message classification
- Displays interactive data visualizations
- Includes user feedback system for model improvement
- Features recommendation engine for disaster response

---

## ETL Pipeline

**Key Features:**
- Comprehensive input validation and error handling
- Advanced text preprocessing with NLTK
- Efficient data cleaning using pandas operations
- SQLite database storage with proper indexing

**Data Processing Steps:**
1. Load and validate input CSV files
2. Merge messages and categories datasets
3. Split category strings into binary columns
4. Handle edge cases and data inconsistencies
5. Remove duplicates and summarize results
6. Save cleaned data to SQLite database

**Command Usage:**
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

---

## Machine Learning Pipeline

**Advanced Features:**
- Custom tokenization with lemmatization and URL detection
- TF-IDF vectorization with preprocessed stop words
- RandomForestClassifier with balanced class weights
- Comprehensive hyperparameter tuning
- Multi-output classification for 36 categories

**Key Components:**
- **Text Processing:** Custom tokenizer with NLTK lemmatization
- **Feature Extraction:** TF-IDF with optimized parameters
- **Model:** RandomForest with class balancing
- **Evaluation:** Per-category metrics and confusion matrices
- **Output:** Trained model saved as pickle file

**Hyperparameters Tuned:**
- `n_estimators`: [50, 100]
- `max_depth`: [None, 10, 20]
- `min_samples_split`: [2, 4]
- `max_features`: ['sqrt', 'log2']

**Command Usage:**
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

---

## Flask Web Application

**Advanced Features:**
- Interactive message classification interface
- 5 sophisticated data visualizations using Plotly
- User feedback system with SQLAlchemy database
- Recommendation engine for disaster response
- Performance optimizations with caching and limiting

**Technical Implementation:**
- Flask framework with SQLAlchemy integration
- Model caching for improved performance
- Dynamic category detection
- Responsive design for mobile compatibility

---

## Visualizations

The web application includes five comprehensive visualizations:

1. **Category Distribution Bar Chart** - Shows frequency of each disaster category
2. **Time Series Analysis** - Messages over time grouped by genre
3. **Category Correlation Heatmap** - Relationships between different categories
4. **Message Length Distribution** - Histogram of message character lengths
5. **Top Words Frequency Chart** - Most common words across all messages

**Future Visualization Ideas:**
- Word clouds for category-specific terms
- Geographic distribution of messages
- Real-time classification confidence scores
- Category prediction trends over time

---

## Model Performance

### Sample Model Metrics:

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| medical_help | 0.75 | 0.60 | 0.67 | 150 |
| water | 0.81 | 0.35 | 0.49 | 120 |
| shelter | 0.70 | 0.52 | 0.60 | 200 |
| food | 0.68 | 0.45 | 0.54 | 180 |
| clothing | 0.85 | 0.25 | 0.38 | 45 |
| search_and_rescue | 0.90 | 0.30 | 0.45 | 60 |

**Overall Performance:**
- **Weighted F1-Score:** 0.72
- **Average Precision:** 0.74
- **Average Recall:** 0.46

**Key Insights:**
- High precision across most categories indicates low false positive rates
- Lower recall for minority classes reflects dataset imbalance challenges
- Model performs best on categories with sufficient training examples

---

## Imbalanced Dataset Discussion

### Challenge Analysis
Some labels have thousands of examples while others are severely underrepresented:
- `aid_related`: ~8,000 examples
- `water`: ~400 examples  
- `search_and_rescue`: ~180 examples

### Impact on Model Performance
- **High Accuracy Paradox:** Model achieves high overall accuracy but poor recall for rare categories
- **Critical Miss Risk:** In disaster scenarios, missing urgent needs is more costly than false alarms
- **Evaluation Complexity:** Standard accuracy metrics can be misleading

### Mitigation Strategies Implemented
1. **Balanced Class Weights:** `class_weight='balanced'` in RandomForestClassifier
2. **Comprehensive Metrics:** Focus on F1-score, precision, and recall per category
3. **Confusion Matrix Analysis:** Visual assessment of per-category performance
4. **Weighted Scoring:** GridSearchCV uses weighted F1-score for optimization

### Future Improvements
- **Oversampling Techniques:** SMOTE or ADASYN for minority classes
- **Ensemble Methods:** Combine multiple models with different strengths
- **Cost-Sensitive Learning:** Assign different misclassification costs
- **Deep Learning:** Neural networks with focal loss for imbalanced data

---

## Recommendations System

Based on classification results, the application provides actionable recommendations for contacting relevant disaster response organizations:

| Category | Recommended Organizations | Action Items |
|----------|---------------------------|--------------|
| water | Water.org, UNICEF, Red Cross | Coordinate water supply delivery, purification systems |
| food | World Food Programme, local food banks | Organize food distribution, nutritional support |
| shelter | UNHCR, Red Cross, Habitat for Humanity | Deploy temporary housing, coordinate shelter assignments |
| medical_help | Doctors Without Borders, Red Cross | Medical team deployment, supply coordination |
| search_and_rescue | Local emergency services, FEMA | Search operations, rescue coordination |
| security | UN Peacekeeping, local authorities | Security assessment, protection services |
| clothing | Salvation Army, local charities | Clothing drive coordination, distribution |
| money | Financial aid organizations | Emergency financial assistance programs |

### Implementation Features
- **Context-Aware Suggestions:** Recommendations adapt based on predicted categories
- **Priority Ranking:** Multiple categories trigger prioritized response suggestions
- **Contact Information:** Direct links to organization websites and contact details
- **Resource Allocation:** Guidance on resource deployment and coordination

---

## Instructions

### Prerequisites
Ensure you have Python 3.7+ installed with the required dependencies.

### 1. Setting Up the Database and Model

Run the following commands in the project's root directory:

**Step 1: ETL Pipeline**
```bash
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```

**Step 2: Machine Learning Pipeline**
```bash
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```

**Step 3: Verify Setup**
Ensure the following files are created:
- `data/DisasterResponse.db` (SQLite database)
- `models/classifier.pkl` (Trained model)

### 2. Running the Web Application

Navigate to the app directory and start the Flask server:
```bash
cd app
python run.py
```

**Access the application at:** [http://localhost:3001](http://localhost:3001)

### 3. Using the Application
1. **Home Page:** View data visualizations and browse message samples
2. **Classification:** Enter a disaster message to see predicted categories
3. **Recommendations:** Review suggested response actions based on predictions
4. **Feedback:** Provide feedback on classification accuracy to improve the model

---

## Web App Features

### Core Functionality
1. **Real-Time Message Classification**
   - Input disaster-related messages via web interface
   - Instant multi-category classification results
   - Confidence scores and probability distributions

2. **Interactive Data Visualizations**
   - Dynamic charts built with Plotly
   - Responsive design for various screen sizes
   - Real-time data filtering and exploration

3. **User Feedback System**
   - Rate classification accuracy
   - Submit corrections for misclassified messages  
   - Database storage for continuous model improvement

4. **Recommendation Engine**
   - Context-aware disaster response suggestions
   - Direct links to relevant organizations
   - Priority-based action item lists

### Performance Features
- **Model Caching:** LRU cache for improved response times
- **Configurable Limits:** User-selectable table row limits
- **Lazy Loading:** Efficient data loading for large datasets
- **Mobile Responsive:** Optimized for mobile and tablet devices

### Data Management
- **Searchable Tables:** Filter messages by category or content
- **Export Functionality:** Download classification reports
- **Data Visualization:** Multiple chart types for insights
- **Historical Tracking:** View classification trends over time

---

## Deployment

The application is designed for deployment on various cloud platforms:

### Recommended Platforms
1. **Render** (Recommended)
   - Simple deployment process
   - Free tier available
   - Automatic HTTPS
   
2. **Heroku**
   - Git-based deployment
   - Add-on ecosystem
   - Scalable infrastructure

3. **AWS EC2**
   - Full control over environment
   - Scalable compute resources
   - Integration with AWS services

4. **Railway**
   - Modern deployment platform
   - Simple configuration
   - Built-in monitoring

### Deployment Checklist
- [ ] Update database and model file paths for production
- [ ] Configure environment variables for sensitive data
- [ ] Set up proper logging and error handling
- [ ] Implement SSL/TLS certificates
- [ ] Configure auto-scaling and monitoring
- [ ] Test all functionality in production environment

### Environment Variables
```bash
DATABASE_URL=sqlite:///DisasterResponse.db
MODEL_PATH=models/classifier.pkl
SECRET_KEY=your-secret-key-here
DEBUG=False
```

---

## Installation

### System Requirements
- Python 3.7 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
- **Flask 2.3.2** - Web framework
- **scikit-learn 1.3.0** - Machine learning library
- **pandas 2.0.3** - Data manipulation
- **nltk 3.8.1** - Natural language processing
- **SQLAlchemy 2.0.19** - Database ORM
- **plotly 5.15.0** - Interactive visualizations
- **joblib 1.3.2** - Model serialization
- **numpy 1.24.3** - Numerical computing

### Development Dependencies
- **pytest** - Testing framework
- **flake8** - Code linting
- **black** - Code formatting
- **jupyter** - Notebook development

### Virtual Environment Setup
```bash
python -m venv disaster-response
source disaster-response/bin/activate  # On Windows: disaster-response\Scripts\activate
pip install -r requirements.txt
```

---

## File Structure

```
disaster-response-pipeline/
├── app/
│   ├── templates/
│   │   ├── master.html          # Main dashboard page
│   │   └── go.html              # Classification results page
│   └── run.py                   # Flask application server
├── data/
│   ├── disaster_categories.csv  # Raw categories data
│   ├── disaster_messages.csv    # Raw messages data
│   ├── process_data.py          # ETL pipeline script
│   └── DisasterResponse.db      # Processed SQLite database
├── models/
│   ├── train_classifier.py     # ML pipeline script
│   └── classifier.pkl          # Trained model file
├── notebooks/                  # Jupyter notebooks for exploration
│   ├── ETL Pipeline Preparation.ipynb
│   └── ML Pipeline Preparation.ipynb
├── screenshots/                # Application screenshots
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── .gitignore                 # Git ignore rules
```

### Key Files Description
- **`run.py`**: Main Flask application with routing and visualization logic
- **`process_data.py`**: ETL pipeline for data cleaning and database creation
- **`train_classifier.py`**: ML pipeline for model training and evaluation
- **`master.html`**: Main web interface template
- **`go.html`**: Classification results display template

---

## Screenshots

### Dashboard Interface
*Main dashboard showing data visualizations and message browser*

### Classification Results
*Example message classification with predicted categories and recommendations*

### Performance Metrics
*Model evaluation dashboard with confusion matrices and performance charts*

*Note: Screenshots will be added upon completion of final testing*

---

## License and Acknowledgements

### License
This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgements
- **Udacity Data Science Nanodegree Program** - Project framework and guidance
- **Figure Eight (Appen)** - Disaster response dataset
- **Flask Community** - Web framework and documentation
- **Scikit-learn Contributors** - Machine learning library
- **Plotly Team** - Interactive visualization tools

### Data Source
The disaster response dataset was provided by Figure Eight (formerly CrowdFlower) and contains real messages from various disaster events. This dataset is used for educational purposes as part of the Udacity Data Science Nanodegree program.

### Contributing
This project was developed as part of academic coursework. While not actively maintained, feedback and suggestions are welcome for educational purposes.

---

**Project Completed:** August 2025  
**Udacity Data Science Nanodegree**  
**Disaster Response Pipeline Project**