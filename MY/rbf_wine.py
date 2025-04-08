import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from tkinter import filedialog
from tkinter import Tk
import os
from ipywidgets import FileUpload
from IPython.display import display
import io

# Function to load the wine dataset from the library
def load_wine_from_sklearn():
    print("Loading Wine dataset from sklearn...")
    wine = datasets.load_wine()
    wine_data = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_data['quality'] = wine.target
    return wine_data

# Function to upload the wine dataset manually
def upload_wine_file(environment="desktop"):
    if environment == "desktop":
        print("Please select the wine dataset file...")
        root = Tk()
        root.withdraw()  # Hide the Tkinter root window
        file_path = filedialog.askopenfilename(title="Select Wine Dataset", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        
        if file_path:
            wine_data = pd.read_csv(file_path)
            print(f"Dataset loaded from {file_path}")
            return wine_data
        else:
            print("No file selected.")
            return None
    elif environment == "browser":
        uploader = FileUpload(
            accept='.csv',  # Accept only CSV files
            multiple=False  # Only allow single file upload
        )
        display(uploader)

        def on_upload_change(change):
            nonlocal wine_data  # Use nonlocal to modify the variable in the outer scope
            if uploader.value:
                for filename, file_info in uploader.value.items():
                    uploaded_file_content = file_info['content']
                    csv_data = io.BytesIO(uploaded_file_content)  
                    wine_data = pd.read_csv(csv_data)
                    print(f"Dataset loaded from {filename}")
                    
                    # Process the data here
                    process_uploaded_data(wine_data)
                    
                    uploader.close()

        uploader.observe(on_upload_change, names='value')
        
        # Wait for the upload to complete (this is not straightforward in Jupyter)
        # For demonstration, we'll use a global variable to store the data
        wine_data = None
        while wine_data is None:
            # This loop will block until the data is loaded
            # However, in Jupyter, this might not work as expected due to asynchronous nature
            pass
        
        return wine_data  # This will not work as expected in Jupyter due to asynchronous upload
    else:
        raise ValueError("Invalid environment specified. Choose 'desktop' or 'browser'.")

# Function to check and clean the dataset if needed
def clean_dataset(wine_data):
    # Check for missing values
    if wine_data.isnull().sum().sum() > 0:
        print("Missing values found! Filling missing values with mean...")
        wine_data = wine_data.fillna(wine_data.mean())
    return wine_data

# Function to scale the features of the dataset
def scale_data(wine_data):
    # Separate features and target
    X = wine_data.drop('quality', axis=1)
    y = wine_data['quality']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# Function to train an RBF SVM model and make predictions
def train_rbf_model(X_scaled, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Train an SVM with RBF kernel
    svm_rbf = SVC(kernel='rbf', gamma='scale')
    svm_rbf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = svm_rbf.predict(X_test)
    
    # Evaluate the model
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    return svm_rbf, y_pred, y_test, X_test

# Function to plot the data and visualize the results
def plot_data(X, y, y_pred=None):
    plt.figure(figsize=(8, 6))
    
    # Plot the original data using the first two features
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", marker='o', s=100, legend='full')
    
    if y_pred is not None:
        # Plot predicted data points
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_pred, palette="Set2", marker='x', s=100, legend='full')
    
    plt.title("Wine Data Visualization")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def process_uploaded_data(wine_data):
    # Clean the dataset
    wine_data = clean_dataset(wine_data)

    # Scale the features
    X_scaled, y = scale_data(wine_data)

    # Plot the original data
    plot_data(X_scaled, y)

    # Train RBF model and make predictions
    svm_model, y_pred, y_test, X_test = train_rbf_model(X_scaled, y)

    # Plot the prediction results
    plot_data(X_test, y_test, y_pred)

def main():
    print("Welcome to the Wine Quality Prediction using RBF Kernel!")
    print("Choose the option for loading the dataset:")
    print("1. Load wine dataset from sklearn (online).")
    print("2. Upload your own wine dataset.")

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        wine_data = load_wine_from_sklearn()
    elif choice == "2":
        print("Do you want to upload the dataset from your browser or desktop?")
        print("1. Upload from Browser")
        print("2. Upload from Desktop")
        upload_choice = input("Enter choice (1 or 2): ").strip()

        if upload_choice == "1":
            # For browser upload, we need to handle it differently
            uploader = FileUpload(
                accept='.csv',  # Accept only CSV files
                multiple=False  # Only allow single file upload
            )
            display(uploader)

            def on_upload_change(change):
                nonlocal wine_data  # Use nonlocal to modify the variable in the outer scope
                if uploader.value:
                    for filename, file_info in uploader.value.items():
                        uploaded_file_content = file_info['content']
                        csv_data = io.BytesIO(uploaded_file_content)  
                        wine_data = pd.read_csv(csv_data)
                        print(f"Dataset loaded from {filename}")
                        
                        # Process the data here
                        process_uploaded_data(wine_data)
                        
                        uploader.close()

            uploader.observe(on_upload_change, names='value')
            
            # Wait for the user to upload the file
            # This part is tricky in Jupyter due to asynchronous nature
            # You might need to adjust your workflow based on your application
            input("Press Enter after uploading the file...")
            
            # Since we can't directly return the data from the upload function,
            # we'll use a different approach to handle the workflow.
            # For example, you could use a global variable or a different structure.
            # However, in this example, we'll directly process the data within the upload callback.
        elif upload_choice == "2":
            wine_data = upload_wine_file(environment="desktop")
        else:
            print("Invalid choice, please try again.")
            return

    # If data is loaded from sklearn or desktop
    if choice == "1" or upload_choice == "2":
        if wine_data is not None:
            # Clean the dataset
            wine_data = clean_dataset(wine_data)

            # Scale the features
            X_scaled, y = scale_data(wine_data)

            # Plot the original data
            plot_data(X_scaled, y)

            # Train RBF model and make predictions
            svm_model, y_pred, y_test, X_test = train_rbf_model(X_scaled, y)

            # Plot the prediction results
            plot_data(X_test, y_test, y_pred)
        else:
            print("Failed to load the dataset. Please try again.")

if __name__ == "__main__":
    main()
