import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def assess_model(model, X_test, y_test):
    # Measure model accuracy on DB2 test split
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

    # Get unique class labels
    classes = np.unique(y_test)

    # Calculate total and correct predictions per class
    summary_data = []
    for label in classes:
        total = np.sum(y_test == label)
        correct = np.sum((y_test == label) & (y_pred == label))
        summary_data.append({
            'Class': label,
            'Total Samples': total,
            'Correctly Classified': correct,
            'Accuracy (%)': round(100 * correct / total, 2)
        })

    # Display results as a DataFrame
    summary_df = pd.DataFrame(summary_data)
    print("\nPer-Class Classification Summary:")
    print(summary_df.to_string(index=False))


def assess_model_nn(encoder, gesture_decoder, recon_decoder, X_test, y_test):
    encoder.eval()
    gesture_decoder.eval()
    if recon_decoder:
        recon_decoder.eval()

    with torch.no_grad():
        # Run inference
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        z = encoder(X_test_tensor)
        logits = gesture_decoder(z)
        if recon_decoder:
            recon = recon_decoder(z)

        # Classification
        y_pred = torch.argmax(logits, dim=1).numpy()
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nClassification Accuracy: {accuracy:.4f}")
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix")
        plt.show()

        # Per-class breakdown
        classes = np.unique(y_test)
        summary_data = []
        for label in classes:
            total = np.sum(y_test == label)
            correct = np.sum((y_test == label) & (y_pred == label))
            summary_data.append({
                'Class': label,
                'Total Samples': total,
                'Correctly Classified': correct,
                'Accuracy (%)': round(100 * correct / total, 2)
            })

        summary_df = pd.DataFrame(summary_data)
        print("\nPer-Class Classification Summary:")
        print(summary_df.to_string(index=False))

        # Reconstruction quality
        if recon_decoder:
            recon_np = recon.numpy()
            mse = np.mean((X_test - recon_np) ** 2)
            print(f"\nReconstruction MSE: {mse:.6f}")
