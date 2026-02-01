import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def create_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Creates a transfer learning model using ResNet50 for brain tumor classification.
    Unfreezes the last 20 layers for fine-tuning on MRI images.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of tumor classes to predict.

    Returns:
        tf.keras.Model: The compiled model.
    """
    # Load ResNet50 pretrained on ImageNet, without the top classification layer
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze all layers except the last 20 — allows fine-tuning on MRI domain
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    # Build classification head on top
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Lower learning rate to avoid destroying pretrained weights during fine-tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()
