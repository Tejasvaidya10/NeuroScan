import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocesses an image for the model.

    Args:
        image (PIL.Image.Image): The input image.
        target_size (tuple): The target size to resize the image to.

    Returns:
        np.array: The preprocessed image tensor ready for prediction.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)  # ResNet50 ImageNet preprocessing

    return image_array

def load_trained_model(model_path):
    """
    Loads a trained Keras model from the specified path.

    Args:
        model_path (str): Path to the saved .h5 or .keras model file.

    Returns:
        tf.keras.Model: The loaded model, or None if loading fails.
    """
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except (FileNotFoundError, OSError) as e:
        print(f"Error loading model: {e}")
        return None
