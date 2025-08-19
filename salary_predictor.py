import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import tkinter as tk
from tkinter import messagebox, ttk
import joblib

# Load dataset
data = pd.read_csv("Salary_Data.csv")
data = data.dropna(subset=['Salary'])

# Encode categorical columns
categorical_cols = ['Gender', 'Education Level', 'Job Title']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Define features and target
X = data.drop('Salary', axis=1)
y = data['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Evaluation:")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")

# Save trained model
joblib.dump(model, "salary_model.pkl")

# ------------------------ GUI -----------------------------

def predict_salary():
    try:
        age = int(entry_age.get())
        experience = float(entry_exp.get())
        gender = gender_var.get()
        education = education_var.get()
        job_title = job_var.get()

        input_data = {
            'Age': [age],
            'Years of Experience': [experience]
        }

        for col in X.columns:
            if col.startswith("Gender_"):
                input_data[col] = [1 if col == f"Gender_{gender}" else 0]
            elif col.startswith("Education Level_"):
                input_data[col] = [1 if col == f"Education Level_{education}" else 0]
            elif col.startswith("Job Title_"):
                input_data[col] = [1 if col == f"Job Title_{job_title}" else 0]

        for col in X.columns:
            if col not in input_data:
                input_data[col] = [0]

        input_df = pd.DataFrame(input_data)
        model = joblib.load("salary_model.pkl")
        prediction = model.predict(input_df)[0]
        result_label.config(text=f"Predicted Salary: ₹{int(prediction):,}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Setup GUI
app = tk.Tk()
app.title("💼 Salary Predictor")
app.geometry("460x570")
app.configure(bg="#e6f2ff")  # light blue background

style = ttk.Style()
style.theme_use("clam")
style.configure("TLabel", font=("Arial", 12), background="#e6f2ff", foreground="#003366")
style.configure("TButton", font=("Arial", 12, "bold"), background="#004080", foreground="white")
style.configure("TEntry", font=("Arial", 12))
style.configure("TCombobox", font=("Arial", 11))

# Title
title_frame = tk.Frame(app, bg="#004080")
title_frame.pack(fill="x")
tk.Label(title_frame, text="Salary Prediction App", font=("Arial", 18, "bold"), bg="#004080", fg="white", pady=10).pack()

# Input fields
frame = ttk.Frame(app)
frame.pack(pady=20)

labels = ["Age:", "Years of Experience:", "Gender:", "Education Level:", "Job Title:"]

entry_age = ttk.Entry(frame)
entry_exp = ttk.Entry(frame)

gender_var = tk.StringVar()
gender_combo = ttk.Combobox(frame, textvariable=gender_var, state="readonly")
gender_combo['values'] = ["Male", "Female", "Other"]
gender_combo.current(0)

education_var = tk.StringVar()
education_combo = ttk.Combobox(frame, textvariable=education_var, state="readonly")
education_combo['values'] = ["Bachelors", "Masters", "PhD", "Diploma"]
education_combo.current(0)

job_var = tk.StringVar()
job_combo = ttk.Combobox(frame, textvariable=job_var, state="readonly")
job_combo['values'] = ["Software Engineer", "Data Scientist", "Manager", "Analyst"]
job_combo.current(0)

widgets = [entry_age, entry_exp, gender_combo, education_combo, job_combo]

for i, label in enumerate(labels):
    ttk.Label(frame, text=label).grid(row=i, column=0, padx=10, pady=10, sticky="w")
    widgets[i].grid(row=i, column=1, padx=10, pady=10)

# Predict button
btn_style = {"font": ("Arial", 12, "bold"), "bg": "#007acc", "fg": "white", "padx": 10, "pady": 5}
predict_btn = tk.Button(app, text="Predict Salary", command=predict_salary, **btn_style)
predict_btn.pack(pady=25)

# Result label
result_label = tk.Label(app, text="Predicted Salary: ₹0", font=("Arial", 14, "bold"), bg="#e6f2ff", fg="#006600")
result_label.pack(pady=10)

# Footer
#tk.Label(app, text="ML Project by Krutika", font=("Arial", 10), bg="#e6f2ff", fg="#666666").pack(side="bottom", pady=10)

app.mainloop()
