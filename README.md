# Ridge Regression from Scratch (California Housing Dataset)

This project demonstrates a **complete from-scratch implementation of Ridge Regression** using **pure NumPy**, applied to the **California Housing dataset**.

Instead of relying on `sklearn.linear_model.Ridge`, the goal of this notebook is to deeply understand:

- how regularization works mathematically  
- how weights are updated using Gradient Descent  
- how Ridge Regression helps reduce overfitting  

---

## 📌 Project Objectives

- Implement **Ridge Regression (L2 Regularization)** from scratch
- Apply it on a **real-world regression dataset**
- Understand the effect of **regularization strength (λ / alpha)**
- Evaluate model performance using **R² Score and RMSE**
- Compare train vs test performance to analyze **overfitting control**

---

## 📊 Dataset Used

**California Housing Dataset**

Features include:
- Median income
- House age
- Number of rooms
- Population
- Latitude & longitude
- Other housing-related attributes

**Target Variable**
- `median_house_value`

---

## ⚙️ Workflow

### 1. Data Loading
- Dataset loaded using `sklearn.datasets.fetch_california_housing`
- Converted into NumPy arrays for scratch implementation

### 2. Train–Test Split
- Data split into training and testing sets
- Ensures fair evaluation on unseen data

### 3. Feature Scaling
- Standardization applied manually
- Required because Ridge Regression is **scale-sensitive**

---

## 🧠 Ridge Regression (From Scratch)

### Cost Function Used

\[
J(w, b) = \frac{1}{n} \sum (y - \hat{y})^2 + \lambda \sum w^2
\]

Where:
- λ (alpha) controls regularization strength
- Penalizes large weights
- Helps prevent overfitting

---

### Gradient Descent Updates

**Weight Update**
\[
w := w - \alpha \left( \frac{-2}{n} X^T (y - \hat{y}) + 2\lambda w \right)
\]

**Bias Update**
\[
b := b - \alpha \left( \frac{-2}{n} \sum (y - \hat{y}) \right)
\]

- Bias is **not regularized**
- Only weights are penalized

---

## 🧪 Model Training

- Initialized weights and bias manually
- Trained using Gradient Descent
- Tuned:
  - Learning rate
  - Number of epochs
  - Regularization strength (λ)

---

## 📈 Evaluation Metrics

The model is evaluated using:

- **R² Score**
- **RMSE (Root Mean Squared Error)**

Results are calculated for:
- Training data
- Testing data

This helps identify:
- Overfitting
- Underfitting
- Generalization ability

---

## 🔍 Key Insights

- Ridge Regression reduces overfitting by shrinking weights
- Large λ → higher bias, lower variance
- Small λ → closer to standard Linear Regression
- Feature scaling is mandatory for proper convergence
- From-scratch implementation clarifies what happens inside `.fit()`

---

## 🧰 Tech Stack

- Python
- NumPy
- Matplotlib (for visualization if used)
- Scikit-learn (only for dataset & metrics)

---

## 🚀 Why This Project?

Most ML users write:

```python
Ridge().fit(X, y)
