# Section 1: Importing Libraries
import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import nltk

# Section 2: NLTK Stop Words Setup
nltk.download('stopwords')
stop_words = set(nltk.corpus.stopwords.words('english'))

# Section 3: Global Variables
trained_rf_model = None
trained_vectorizer = None
trained_dt_model = None
df = None
text_entry = None
result_text = None
X_train, X_test, y_train, y_test = None, None, None, None

# Section 4: Functions for Data Preprocessing and Model Prediction
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = text.lower()
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def load_csv():
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        if 'statement' in df.columns and 'status' in df.columns:
            messagebox.showinfo("Success", "CSV loaded successfully.")
        else:
            messagebox.showerror("Error", "CSV must contain 'statement' and 'status' columns.")

def process_data():
    global df, X_train, X_test, y_train, y_test
    if df is None:
        messagebox.showerror("Error", "Please load a CSV file first.")
        return

    df['statement'] = df['statement'].fillna('')
    df = df.dropna(subset=['status'])
    df['processed_text'] = df['statement'].apply(preprocess_text)

    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    global trained_rf_model, trained_vectorizer, trained_dt_model
    trained_rf_model = rf_model
    trained_vectorizer = vectorizer
    trained_dt_model = dt_model

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Data split: 80% for training, 20% for testing.\n")

def predict_rf():
    if trained_rf_model is None or trained_vectorizer is None:
        messagebox.showwarning("Model Error", "Please load data and train the model first.")
        return
    rf_accuracy = trained_rf_model.score(X_test, y_test)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"RandomForest Accuracy on Test Set: {rf_accuracy * 100:.2f}%\n")

def predict_dt():
    if trained_dt_model is None or trained_vectorizer is None:
        messagebox.showwarning("Model Error", "Please load data and train the model first.")
        return
    dt_accuracy = trained_dt_model.score(X_test, y_test)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Decision Tree Accuracy on Test Set: {dt_accuracy * 100:.2f}%\n")

# ✅ Updated Function with Conflict Handling
def predict_statement():
    statement = text_entry.get("1.0", tk.END).strip()
    if not statement:
        messagebox.showwarning("Input Error", "Please enter a statement for prediction.")
        return

    processed_statement = preprocess_text(statement)
    statement_vector = trained_vectorizer.transform([processed_statement])

    rf_prediction = trained_rf_model.predict(statement_vector)[0]
    rf_prob = trained_rf_model.predict_proba(statement_vector)[0]
    rf_pred_prob = max(rf_prob) * 100

    dt_prediction = trained_dt_model.predict(statement_vector)[0]
    dt_prob = trained_dt_model.predict_proba(statement_vector)[0]
    dt_pred_prob = max(dt_prob) * 100

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Prediction for Statement:\n")
    result_text.insert(tk.END, f"RandomForest Prediction: {rf_prediction}, Probability: {rf_pred_prob:.2f}%\n")
    result_text.insert(tk.END, f"DecisionTree Prediction: {dt_prediction}, Probability: {dt_pred_prob:.2f}%\n\n")

    if rf_prediction == dt_prediction:
        result_text.insert(tk.END, f"✅ Final Decision: {rf_prediction} (both models agree)\n")
    else:
        if dt_pred_prob - rf_pred_prob >= 10:
            result_text.insert(tk.END, f"⚠️ Warning: Models disagree. Decision Tree predicts '{dt_prediction}' with higher confidence.\n")
            result_text.insert(tk.END, f"⚖️ Suggested Action: Consider both predictions and review manually.\n")
        else:
            result_text.insert(tk.END, f"⚠️ Models disagree. Defaulting to RandomForest prediction: {rf_prediction}\n")

# Section 5: Tkinter GUI Setup
def create_gui():
    global text_entry, result_text
    root = tk.Tk()
    root.title("Mental Health Disorder Detection")
    root.geometry("800x600")
    root.config(bg="#9B1B30")

    title_label = tk.Label(root, 
                           text="Detection and Prediction of Future Mental Disorder from social media data using ML, Ensemble learning and LLM", 
                           font=("Arial", 17, "bold"), 
                           bg="white", 
                           fg="black", 
                           padx=20, 
                           pady=20)
    title_label.pack(pady=20)

    button_frame = tk.Frame(root, bg="#9B1B30")
    button_frame.pack(side=tk.LEFT, padx=20, pady=20, anchor="n")

    load_button = tk.Button(button_frame, text="Load CSV File", font=("Arial", 12), command=load_csv)
    load_button.pack(pady=10, anchor="w")

    train_button = tk.Button(button_frame, text="Train and Test", font=("Arial", 12), command=process_data)
    train_button.pack(pady=10, anchor="w")

    rf_button = tk.Button(button_frame, text="Run RandomForest Algorithm", font=("Arial", 12), command=predict_rf)
    rf_button.pack(pady=10, anchor="w")

    dt_button = tk.Button(button_frame, text="Run Decision Tree Algorithm", font=("Arial", 12), command=predict_dt)
    dt_button.pack(pady=10, anchor="w")

    predict_button = tk.Button(button_frame, text="Predict", font=("Arial", 12), command=predict_statement)
    predict_button.pack(pady=10, anchor="w")

    text_frame = tk.Frame(root, bg="#9B1B30")
    text_frame.pack(side=tk.RIGHT, padx=20, pady=20, anchor="n")

    text_entry_label = tk.Label(text_frame, text="Enter Statement for Prediction:", font=("Arial", 13), bg="#9B1B30", fg="white")
    text_entry_label.pack()

    text_entry = tk.Text(text_frame, height=6, width=50, font=("Arial", 12))
    text_entry.pack(pady=10)

    result_text = tk.Text(text_frame, height=10, width=50, font=("Arial", 12), bg="#f5f5f5")
    result_text.pack(pady=10)

    root.mainloop()

# Main function to initialize GUI
if __name__ == "__main__":
    create_gui()
