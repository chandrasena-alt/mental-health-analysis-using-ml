# mental-health-analysis-using-ml

---
## Detection and Prediction of Future Mental Disorder from Social Media Data
Using Machine Learning, Ensemble Learning & LLMs
This project is a GUI-based Mental Health Disorder Detection System built using Python, Tkinter, Machine Learning Models (Random Forest & Decision Tree), TF-IDF vectorization, and NLP preprocessing techniques.

---
## It allows you to:

- 📂 Upload any dataset containing statement and status columns

- 🧹 Preprocess text using NLP

- 🤖 Train ML models (RF & DT)

- 📊 Check model accuracy

- 🔮 Predict mental health category for any text input

- 🖥️ Use a clean graphical interface for all interactions

  ---

  ## Project Structure
  
  mental-health-detection/
- │── code.py
- │── combined.xlsb- 
- │── smmh.csv
- │── mental_health_statements.csv
- │── Digital Behavior and Mental Health Survey 2022.xlsx

  ---
  ## 🧠 Project Features
#### Machine Learning Models:
- Random Forest Classifier
- decision Tree Classifier

#### NLP Processing:
- Tokenization
- Stopword removal
- Text cleaning
- TF-IDF vectorization

#### GUI Interface:
- Built using Tkinter, featuring:
- Load CSV button
- Train/Test button
- Run Random Forest
- Run Decision Tree
- Predict custom statement
- Real-time results display

#### Dataset Requirements:
- Your CSV file must contain:
- statement → the text input
- status → output category (label)

  ---
  
### 🛠️ Installation

Make sure you have Python installed (>=3.7)

### 📥 Install dependencies
```
pip install pandas scikit-learn nltk torch
```
### Download NLTK stopwords
The script already includes:
```
nltk.download('stopwords')
```
---
### How to Run the Project
Open VS Code or any IDE and run:
```
python code.py
```
---
## ⭐ Conclusion

#### This project demonstrates:
- Efficient NLP preprocessing
- ML model training through a GUI
- Real-time prediction of mental-health related text
- A complete end-to-end ML application

  ---
  

