#pip install tensorflow matplotlib scikit-learn Pillow tk seaborn ipywidgets

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tkinter import Tk, filedialog
from PIL import Image
import os
import io
from ipywidgets import FileUpload
from ipywidgets import FileUpload, Button, VBox, Label
import ipywidgets as widgets
from tkinter import filedialog, Tk
from IPython.display import display

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with validation split for better performance tracking
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Confusion Matrix
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Display confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Define a function to display an image and its prediction
def display_image_and_prediction(image, prediction, actual_label):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.title(f"Prediction: {prediction}, Actual: {actual_label}")
    plt.show()

# Show a sample image and prediction
index = np.random.randint(0, len(x_test))
sample_image = x_test[index]
true_label = y_test[index]
prediction = np.argmax(model.predict(np.expand_dims(sample_image, axis=0)))

display_image_and_prediction(sample_image, prediction, true_label)

def process_image(image_data):
    # This is the function that will process the uploaded image data
    # You can replace this with any functionality you need
    print("Image has been uploaded and is ready for processing.")
    # Here we would do something with `image_data`, like opening or processing it
    return image_data

def choose_image(environment="browser"):
    if environment == "desktop":
        root = Tk()
        root.withdraw()  # Don't need the full tkinter window
        file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        print(f"Selected file path: {file_path}")  # Debugging line
        return file_path
    elif environment == "browser":
        uploader = widgets.FileUpload(
            accept='image/*',  # Accept all image types
            multiple=False  # Only allow single file upload
        )
        display(uploader)

        # This function will be called once the file is uploaded
        def on_upload_change(change):
            # We only process the image when a file is uploaded
            if uploader.value:
                # Loop through the dictionary (even if there's only one file)
                for filename, file_info in uploader.value.items():
                    # Access the uploaded file content directly using the 'content' key
                    uploaded_file_content = file_info['content']
                    
                    # Convert the uploaded file content into an in-memory file
                    image_data = io.BytesIO(uploaded_file_content)  
                    # Call the process_image function with the uploaded image
                    process_image(image_data)
                uploader.close()  # Close the uploader once done

        # Observe the upload value and call the on_upload_change function
        uploader.observe(on_upload_change, names='value')
    else:
        raise ValueError("Invalid environment specified. Choose 'desktop' or 'browser'.")

def process_image(image_data):
    # Open the image using PIL
    img = Image.open(image_data).convert('L')  # Convert to grayscale

    # Resize the image to 28x28 pixels (common size for MNIST)
    img = img.resize((28, 28))

    # Normalize the image to be between 0 and 1
    img_array = np.array(img) / 255.0

    # Example of model prediction (replace with actual model prediction)
    prediction = np.argmax(model.predict(np.expand_dims(img_array, axis=0)))

    # Display the uploaded image and prediction
    display_image_and_prediction(img_array, prediction, "Unknown")

    # Display interactive button to save the model
    #display_save_button(prediction)

def display_image_and_prediction(img_array, prediction, label):
    # This function will display the image and prediction result
    print(f"Prediction: {prediction} | Label: {label}")
    # If you want to display the image using matplotlib (optional)
    # import matplotlib.pyplot as plt
    # plt.imshow(img_array, cmap='gray')
    # plt.show()

def display_save_button(prediction, true_label):
    # Create a button for saving the model if prediction is correct
    button = Button(description="Save Model")
    output = widgets.Output()

    # Define the function to handle button click
    def on_button_click(b):
        if prediction == true_label:
            model.save('mnist_model.h5')
            print("Model saved as mnist_model.h5")
        else:
            print(f"Prediction {prediction} does not match true label {true_label}. Model not saved.")

    button.on_click(on_button_click)
    
    # Display the button and output widget
    display(VBox([button, output]))


# Get user input for environment
environment = input("Are you using this in a 'desktop' or 'browser' environment? ").lower()

# Example usage:
file_path = choose_image(environment=environment)

# If in desktop mode, process the image after choosing it
if environment == "desktop" and file_path:
    with open(file_path, 'rb') as f:
        process_image(f)
