import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import json
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Ensure src package is importable when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import create_model


def train(data_dir, batch_size=32, epochs=50, img_size=(224, 224)):
    """
    Trains the CNN model on the provided dataset.

    Args:
        data_dir (str): Path to the dataset directory (should contain 'Training' and 'Testing' subdirs).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train.
        img_size (tuple): Target size for images.
    """

    # Define paths
    train_dir = os.path.join(data_dir, 'Training')
    test_dir = os.path.join(data_dir, 'Testing')

    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Dataset directories not found at {data_dir}. Expected 'Training' and 'Testing' subfolders.")
        return

    # Data Augmentation for training (with 80/20 validation split)
    # Use ResNet50's preprocessing (ImageNet mean subtraction) instead of 1/255 rescaling
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 80% train, 20% validation from Training folder
    )

    # No augmentation for validation and test
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # Training generator (80% of Training folder)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Validation generator (20% of Training folder)
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Test generator (Testing folder — never seen during training)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Create Model
    num_classes = len(train_generator.class_indices)
    class_names = list(train_generator.class_indices.keys())
    print(f"Detected {num_classes} classes: {train_generator.class_indices}")

    model = create_model(input_shape=img_size + (3,), num_classes=num_classes)

    # Compute class weights to handle imbalanced dataset
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")

    # Callbacks
    checkpoint_dir = 'models'
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_save_path = os.path.join(checkpoint_dir, 'brain_tumor_model.h5')
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001
    )

    # Train
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[checkpoint, reduce_lr],
        class_weight=class_weight_dict
    )

    # Save class labels alongside the model
    labels_path = os.path.join(checkpoint_dir, 'class_labels.json')
    with open(labels_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class labels saved to {labels_path}")

    # Plot Accuracy/Loss
    plot_training_history(history, checkpoint_dir)

    # Evaluate on held-out test set
    evaluate_model(model, test_generator, class_names, checkpoint_dir)

    print(f"Training finished. Model saved to {model_save_path}")


def evaluate_model(model, test_generator, class_names, output_dir):
    """Evaluates the model on the test set and prints a classification report."""
    print("\n" + "="*60)
    print("EVALUATION ON HELD-OUT TEST SET")
    print("="*60)

    # Get predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    # Classification report
    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
    print("\nClassification Report:")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    print("Confusion Matrix:")
    print(cm)

    # Save report to file
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
    print(f"\nEvaluation report saved to {report_path}")


def plot_training_history(history, output_dir='models'):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    save_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Training history plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Brain Tumor Classification Model')
    parser.add_argument('--dataset', type=str, default='dataset', help='Path to the dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    args = parser.parse_args()
    train(args.dataset, args.batch_size, args.epochs)
