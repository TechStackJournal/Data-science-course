# Module 12: Deep Learning Fundamentals

---

## 1. CONCEPT

This module builds enough depth to construct simple neural networks and reason about modern AI systems, without requiring a research-level math background. Equally important: learning when *not* to reach for deep learning, since that judgment call is itself a real skill.

**Perceptrons & backpropagation:**
- A single perceptron computes a weighted sum of inputs, applies a non-linear **activation function**, and produces an output
- Stacking layers of perceptrons (a Multi-Layer Perceptron, MLP) allows the network to learn non-linear combinations of features
- **Backpropagation** is the algorithm that computes how much each weight contributed to the final prediction error, propagating that error backward through the network to update weights via gradient descent — the core mechanism that makes training possible, worth understanding conceptually even without deriving the calculus by hand

**Activation functions:**
- **ReLU** (Rectified Linear Unit) — the modern default for hidden layers; computationally cheap, helps avoid a problem called vanishing gradients that plagued older activation functions
- **Sigmoid** — squashes output to (0, 1), still standard for binary classification output layers
- **Softmax** — generalizes sigmoid to multi-class classification, producing a probability distribution across classes

**Optimizers:**
- **SGD (Stochastic Gradient Descent)** — the foundational optimization algorithm; updates weights using gradients computed on small batches of data
- **Adam** — the modern default optimizer for most tasks; adapts the learning rate per-parameter, generally converges faster and more reliably than plain SGD without much tuning

**Regularization for deep nets:**
- **Dropout** — randomly deactivates a fraction of neurons during training, forcing the network to not over-rely on any single neuron, which reduces overfitting
- **Batch normalization** — normalizes layer inputs during training, which stabilizes and often accelerates training
- **Early stopping** — halts training once validation performance stops improving, preventing the network from continuing to fit training-set noise

**CNNs (Convolutional Neural Networks):**
- Designed for spatial data (primarily images); **convolutions** slide small filters across the input to detect local patterns (edges, textures, then increasingly complex shapes in deeper layers)
- **Pooling** layers downsample the spatial dimensions, reducing computation and providing some translation invariance
- The key insight: convolutions exploit spatial locality (nearby pixels are related) in a way a plain MLP, which treats every pixel as an independent unrelated input, cannot do efficiently

**When to use deep learning vs. classical ML:**
- Deep learning tends to win when: data is unstructured (images, audio, raw text), the dataset is large, and complex non-linear feature interactions matter more than interpretability
- Classical ML (especially gradient-boosted trees) tends to win when: data is tabular/structured, the dataset is small-to-medium, interpretability matters, or training/inference speed and simplicity matter
- Reaching for deep learning by default on tabular business data is a common beginner pattern that usually produces worse results, slower iteration, and less interpretability than a well-tuned XGBoost model — a genuinely important judgment call, not a minor stylistic preference

---

## 2. WHY IT MATTERS

Understanding what's happening inside a neural network — even at a conceptual, non-derivation level — is what lets you debug training problems (a loss that won't decrease, a model that overfits instantly, exploding gradients) rather than treating the whole thing as an opaque box you can only restart and hope improves. It's also increasingly expected baseline knowledge in DS interviews, since so much of the field's public conversation now centers on deep learning-based systems, even for practitioners whose day-to-day work is mostly classical ML.

---

## 3. SYNTAX REFERENCE

```python
# --- Python: PyTorch ---
import torch
import torch.nn as nn
import torch.optim as optim

# Simple MLP for tabular/simple classification
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        return self.layer2(x)

model = SimpleMLP(input_dim=20, hidden_dim=64, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop skeleton
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()        # backpropagation
    optimizer.step()       # weight update

    # Early stopping check (conceptual)
    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(X_val_tensor), y_val_tensor)
    # if val_loss stops improving for N epochs, stop training

# Simple CNN for image classification
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)  # size depends on input image dims

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)
```

```r
# --- R: torch for R (mirrors PyTorch's API closely) ---
library(torch)

simple_mlp <- nn_module(
  initialize = function(input_dim, hidden_dim, output_dim) {
    self$layer1 <- nn_linear(input_dim, hidden_dim)
    self$relu <- nn_relu()
    self$dropout <- nn_dropout(0.3)
    self$layer2 <- nn_linear(hidden_dim, output_dim)
  },
  forward = function(x) {
    x %>% self$layer1() %>% self$relu() %>% self$dropout() %>% self$layer2()
  }
)

model <- simple_mlp(input_dim = 20, hidden_dim = 64, output_dim = 2)
optimizer <- optim_adam(model$parameters, lr = 0.001)
criterion <- nn_cross_entropy_loss()

# Training loop follows the same structure as PyTorch
```

---

## 4. WORKED EXAMPLE

**Scenario:** A manufacturing company wants automated defect detection from product images. Build a CNN classifier, and explicitly justify why deep learning is the right tool here.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- Step 0: Justify the tool choice before writing any model code ---
justification = """
TOOL SELECTION RATIONALE:
- Data type: raw product images — unstructured, spatial data. Classical ML
  would require hand-engineered features (edge detectors, texture descriptors)
  that a CNN learns automatically and typically outperforms.
- Data volume: 50,000 labeled defect/non-defect images available — sufficient
  to train a CNN without severe overfitting, especially with data augmentation.
- Interpretability requirement: moderate, not strict — a defect flag with a
  confidence score is operationally useful even without per-pixel explanation,
  unlike the credit-risk scenario in Module 9 which had a hard regulatory
  interpretability requirement.
CONCLUSION: CNN is justified here. If instead we had 200 images and a hard
requirement for engineers to understand exactly why each defect was flagged,
a classical approach with hand-crafted features might be preferable despite
likely lower raw accuracy.
"""
print(justification)

# --- Step 1: Define the CNN ---
class DefectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 2)  # defect / no-defect

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

model = DefectCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Step 2: Train with early stopping to prevent overfitting ---
best_val_loss = float("inf")
patience, patience_counter = 5, 0

for epoch in range(50):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            val_loss += criterion(model(images), labels).item()
    val_loss /= len(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pt")  # save best checkpoint
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

---

## 5. COMMON MISTAKES

- **Reaching for deep learning on small tabular datasets by default.** As covered in Concept — for most structured business data with a few thousand rows, a tuned gradient-boosted tree will typically outperform a neural network, train faster, and be more interpretable. Deep learning's advantages show up specifically with unstructured data and/or large data volumes.
- **Not using a validation set for early stopping, only training/test.** Without a validation set distinct from the final test set, "early stopping" decisions end up implicitly tuned to the test set, contaminating your final evaluation — maintain a clean three-way split (train/validation/test).
- **Forgetting to normalize/scale input data before training.** Unlike tree-based models, neural networks are sensitive to input scale — unnormalized inputs can cause unstable or very slow training.
- **Using a learning rate that's too high (loss diverges/oscillates) or too low (training is impractically slow).** This is one of the most common practical training problems — if loss isn't decreasing steadily, checking the learning rate is usually the first diagnostic step, before assuming something is architecturally wrong.
- **Ignoring class imbalance in a deep learning classification task.** The same imbalance concerns from Module 7 apply here — a CNN trained on 98% non-defect images without any class weighting or resampling will trivially learn to always predict "no defect."

---

## 6. PRACTICE EXERCISES

**Exercise 1 — Tool Selection**
For each scenario, state whether you'd reach for a classical ML model (e.g., XGBoost) or a deep learning model, and justify your choice:
a) Predicting house prices from 15 structured features (sqft, bedrooms, location, etc.) with 5,000 rows of data
b) Classifying handwritten digits from 60,000 labeled images
c) Predicting customer churn from a structured CRM dataset with 10,000 rows and 30 features

<details>
<summary>Sample Solution</summary>

a) **Classical ML (XGBoost/Random Forest)** — structured tabular data, moderate size, interpretability likely valuable for a real estate stakeholder.
b) **Deep learning (CNN)** — unstructured image data with a large labeled dataset; the canonical use case deep learning excels at.
c) **Classical ML** — structured tabular data at a moderate size; a well-tuned ensemble model will very likely match or beat a neural network here while training faster and staying more interpretable for a business stakeholder asking "why did this customer get flagged."
</details>

---

**Exercise 2 — Diagnosing Training Problems**
You train a neural network and notice the training loss decreases steadily but the validation loss starts increasing after epoch 10, while training loss keeps dropping. What's happening, and what are two techniques from this module that could address it?

<details>
<summary>Hint</summary>
Training loss decreasing while validation loss increases is the classic signature of a specific, very common problem in model training.
</details>

<details>
<summary>Solution</summary>

This is **overfitting** — the model is increasingly fitting patterns specific to the training data (including noise) rather than generalizable patterns, which is why training loss keeps improving while validation performance degrades. Two applicable techniques: (1) **early stopping** — stop training at (or restore to) the epoch with the best validation loss, roughly epoch 10 here; (2) **dropout** — add or increase dropout regularization to reduce the network's tendency to over-rely on specific neurons/patterns from the training data. Reducing model complexity (fewer layers/parameters) or adding more training data are also valid approaches.
</details>

---

**Exercise 3 — Full CNN with Justified Tool Choice (Applied)**
Pick an image classification dataset (a public dataset like CIFAR-10 or a domain-relevant one if you can find one) and build a CNN classifier following the worked example's structure: write an explicit tool-selection justification first, build the CNN with dropout and early stopping, and evaluate on a held-out test set. As a deliberate contrast exercise, also note what a classical ML approach on the same problem would require (e.g., manual feature extraction) to reinforce why the deep learning choice was justified here specifically.
