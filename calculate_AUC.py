import torch
from sklearn.metrics import roc_auc_score
import numpy as np


# Assuming you have your test_loader and your model loaded...
def calculate_test_metrics(model, test_loader, device):
    model.eval()

    all_true_labels = []
    all_pred_probs = []

    print("[System] Running Inference for AUC Calculation...")

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            # Forward pass
            logits = model(inputs)

            # Get the probability (confidence score from 0.0 to 1.0)
            probs = torch.sigmoid(logits).cpu().numpy()

            # Store them
            all_pred_probs.extend(probs)
            all_true_labels.extend(labels.numpy())

    # Calculate AUC
    auc_score = roc_auc_score(all_true_labels, all_pred_probs)

    print(f"=====================================")
    print(f"Final Test AUC Score: {auc_score:.4f}")
    print(f"=====================================")

    return auc_score