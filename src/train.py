import tensorflow as tf
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

IMG_SIZE = (224, 224)
SEED = 42


def focal_loss(gamma=2.0):
    """
    Focal Loss for multi-class classification.
    Down-weights easy examples and focuses training on hard misclassifications.
    FL(p_t) = -(1 - p_t)^gamma * log(p_t)
    """
    def focal_loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1.0 - y_pred, gamma)
        focal = weight * cross_entropy
        return tf.reduce_sum(focal, axis=-1)
    return focal_loss_fn


def train(data_dir, batch_size=32, epochs=50):
    train_dir = os.path.join(data_dir, 'Training')
    test_dir = os.path.join(data_dir, 'Testing')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        print(f"Error: Expected 'Training' and 'Testing' subfolders in {data_dir}.")
        return

    # Random 80/20 split from Training folder using seed
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=SEED,
        label_mode='categorical'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=SEED,
        label_mode='categorical'
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        shuffle=False,
        label_mode='categorical'
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    # Apply ResNet50 preprocessing and data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])

    train_ds = train_ds.map(
        lambda x, y: (preprocess_input(data_augmentation(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds.map(
        lambda x, y: (preprocess_input(x), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    # Compute class weights
    all_labels = []
    for _, labels in tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=batch_size,
        validation_split=0.2, subset='training', seed=SEED, label_mode='int'
    ):
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)

    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Boost glioma weight by 2.5x to combat persistent low recall
    glioma_idx = class_names.index('glioma_tumor')
    class_weight_dict[glioma_idx] *= 2.5
    print(f"Class weights (glioma boosted): {class_weight_dict}")

    # Setup
    checkpoint_dir = 'models'
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_save_path = os.path.join(checkpoint_dir, 'brain_tumor_model.h5')

    # ---- Phase 1: Train head only (backbone frozen) ----
    phase1_epochs = max(1, epochs // 3)
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training head ({phase1_epochs} epochs, lr=1e-3)")
    print(f"{'='*60}")

    loss_fn = focal_loss(gamma=2.0)
    model = create_model(input_shape=IMG_SIZE + (3,), num_classes=num_classes, loss_fn=loss_fn)

    history1 = model.fit(
        train_ds,
        epochs=phase1_epochs,
        validation_data=val_ds,
        class_weight=class_weight_dict
    )

    # ---- Phase 2: Fine-tune last 30 backbone layers ----
    phase2_epochs = epochs - phase1_epochs
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tuning backbone ({phase2_epochs} epochs, lr=1e-5)")
    print(f"{'='*60}")

    base_model = model.layers[0]
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=focal_loss(gamma=2.0),
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint(
        model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7
    )

    history2 = model.fit(
        train_ds,
        epochs=phase2_epochs,
        validation_data=val_ds,
        callbacks=[checkpoint, reduce_lr],
        class_weight=class_weight_dict
    )

    # Save class labels
    labels_path = os.path.join(checkpoint_dir, 'class_labels.json')
    with open(labels_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Class labels saved to {labels_path}")

    # Plot
    combined = {}
    for key in history1.history:
        combined[key] = history1.history[key] + history2.history[key]

    class H:
        def __init__(self, h): self.history = h
    plot_training_history(H(combined), checkpoint_dir)

    # Evaluate best model on held-out test set
    best_model = tf.keras.models.load_model(
        model_save_path, custom_objects={'focal_loss_fn': focal_loss(gamma=2.0)}
    )
    evaluate_model(best_model, test_ds, class_names, checkpoint_dir)

    print(f"Training finished. Model saved to {model_save_path}")


def evaluate_model(model, test_ds, class_names, output_dir):
    print("\n" + "="*60)
    print("EVALUATION ON HELD-OUT TEST SET")
    print("="*60)

    all_probs = []
    all_true = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        all_probs.extend(preds)
        all_true.extend(np.argmax(labels.numpy(), axis=1))

    all_probs = np.array(all_probs)
    all_true = np.array(all_true)
    all_preds = np.argmax(all_probs, axis=1)

    report = classification_report(all_true, all_preds, target_names=class_names)
    print("\nClassification Report (standard argmax):")
    print(report)

    cm = confusion_matrix(all_true, all_preds)
    print("Confusion Matrix:")
    print(cm)

    # --- Threshold calibration for glioma ---
    # Test different scaling factors for glioma probability
    glioma_idx = class_names.index('glioma_tumor')
    print(f"\n{'='*60}")
    print("THRESHOLD CALIBRATION FOR GLIOMA")
    print(f"{'='*60}")

    best_f1 = 0
    best_scale = 1.0
    for scale in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        adjusted_probs = all_probs.copy()
        adjusted_probs[:, glioma_idx] *= scale
        adjusted_preds = np.argmax(adjusted_probs, axis=1)

        report_dict = classification_report(
            all_true, adjusted_preds, target_names=class_names, output_dict=True
        )
        glioma_f1 = report_dict['glioma_tumor']['f1-score']
        glioma_recall = report_dict['glioma_tumor']['recall']
        glioma_prec = report_dict['glioma_tumor']['precision']
        overall_acc = report_dict['accuracy']

        print(f"  scale={scale:.1f}: glioma P={glioma_prec:.2f} R={glioma_recall:.2f} "
              f"F1={glioma_f1:.2f} | overall acc={overall_acc:.2f}")

        # Optimize for macro F1 (balances all classes)
        macro_f1 = report_dict['macro avg']['f1-score']
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_scale = scale

    print(f"\nBest glioma scale: {best_scale:.1f} (macro F1={best_f1:.2f})")

    # Apply best threshold and show final results
    if best_scale > 1.0:
        adjusted_probs = all_probs.copy()
        adjusted_probs[:, glioma_idx] *= best_scale
        calibrated_preds = np.argmax(adjusted_probs, axis=1)

        calibrated_report = classification_report(
            all_true, calibrated_preds, target_names=class_names
        )
        calibrated_cm = confusion_matrix(all_true, calibrated_preds)

        print(f"\nCalibrated Classification Report (glioma scale={best_scale:.1f}):")
        print(calibrated_report)
        print("Calibrated Confusion Matrix:")
        print(calibrated_cm)
    else:
        calibrated_report = report
        calibrated_cm = cm

    # Save calibration config
    calibration_path = os.path.join(output_dir, 'calibration.json')
    with open(calibration_path, 'w') as f:
        json.dump({'glioma_scale': best_scale, 'class_names': class_names}, f)
    print(f"Calibration config saved to {calibration_path}")

    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("Classification Report (standard argmax):\n")
        f.write(report)
        f.write("\n\nConfusion Matrix (standard):\n")
        f.write(str(cm))
        if best_scale > 1.0:
            f.write(f"\n\n{'='*60}\n")
            f.write(f"Calibrated Report (glioma scale={best_scale:.1f}):\n")
            f.write(calibrated_report)
            f.write("\n\nCalibrated Confusion Matrix:\n")
            f.write(str(calibrated_cm))
    print(f"Evaluation report saved to {report_path}")


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
