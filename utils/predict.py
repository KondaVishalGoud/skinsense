from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained stacking model
model = load_model('model/stacking.keras')

def predict_disease(img_file):
    """
    Predict the disease based on the input image file.
    
    Parameters:
    img_file (file-like object): The uploaded image file.
    
    Returns:
    str: The predicted disease class.
    """
    # Preprocess the image
    img_array = preprocess_image(img_file)
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Apply softmax to the prediction
    prediction = tf.nn.softmax(prediction).numpy()
    
    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)
    
    # Define the disease classes
    disease_classes = ['1. Eczema', '2. Melanoma', '3. Atopic Dermatitis', '4. Basal Cell Carcinoma', '5. Melanocytic Nevi (NV)',
                       '6. Benign Keratosis-like Lesions (BKL)', '7. Psoriasis', '8. Seborrheic Keratoses', '9. Tinea Ringworm Candidiasis', '10. Warts Molluscum']
    
    # Get the result
    result = disease_classes[predicted_class[0]]
    
    return result
