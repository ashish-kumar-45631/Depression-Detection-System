import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tkinter as tk
from tkinter import messagebox

# Load the dataset
data = pd.read_csv('depression.csv')

# Features and labels
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize the model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)


# Create the GUI application
root = tk.Tk()
root.title("Depression Detection System")

# Function to detect depression
def detect_depression():
    # Get user input from GUI fields
    inputs = [int(entry.get()) for entry in entry_fields]
    
    # Predict using the trained model
    prediction = model.predict([inputs])
    
    # Show the result in a message box
    if prediction[0] == 1:
        messagebox.showinfo("Result", "You may be suffering from depression. Please consult a healthcare professional.")
    else:
        messagebox.showinfo("Result", "You are not showing signs of depression. However, if you have concerns, consult a healthcare professional.")

# Create GUI input fields
labels = ['Sleep hours', 'Social interaction score', 'Physical activity score', 'Stress level']
entry_fields = []
for i, label_text in enumerate(labels):
    label = tk.Label(root, text=label_text)
    label.grid(row=i, column=0)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1)
    entry_fields.append(entry)

# Create the detect button
detect_button = tk.Button(root, text="Detect Depression", command=detect_depression)
detect_button.grid(row=len(labels), columnspan=2)

# Run the GUI application
root.mainloop()
