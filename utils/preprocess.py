from PIL import Image
import numpy as np
import tensorflow as tf

def preprocess_image(img_file):
    """
    Preprocess the input image file for prediction.
    
    Parameters:
    img_file (file-like object): The uploaded image file.
    
    Returns:
    np.array: Preprocessed image ready for model prediction.
    """
    # Load the image using PIL
    img = Image.open(img_file).convert('RGB')
    
    # Resize the image to match model input
    img = img.resize((224, 224))
    
    # Convert image to array
    img_array = np.array(img)
    
    # Expand dimensions to match model input
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image for ResNet50 or other models
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    return img_array
