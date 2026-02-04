import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

def create_model(input_shape=(224, 224, 3), num_classes=4, loss_fn='categorical_crossentropy'):
    """
    Creates a transfer learning model using ResNet50 for brain tumor classification.
    Uses a two-stage approach: frozen backbone first, then fine-tuning.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).
        num_classes (int): The number of tumor classes to predict.
        loss_fn: Loss function (string or callable).

    Returns:
        tf.keras.Model: The compiled model.
    """
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    # Freeze backbone initially
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=loss_fn,
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    model = create_model()
    model.summary()
