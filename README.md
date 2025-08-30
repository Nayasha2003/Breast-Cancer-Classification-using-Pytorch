🩺 Breast Cancer Classification using PyTorch

A deep learning project for binary classification of breast cancer tumors (benign vs malignant) using the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn. Implemented with PyTorch.

📊 Dataset

Source: sklearn.datasets.load_breast_cancer

Samples: 569

Features: 30 real-valued features (e.g., radius, texture, smoothness, compactness, etc.)

Classes:

0: Malignant (cancerous)

1: Benign (non-cancerous)

🧠 Model Architecture

Input Layer: 30 neurons (one per feature)

Hidden Layer: 64 neurons, ReLU activation

Output Layer: 1 neuron, Sigmoid activation

Loss: Binary Cross-Entropy Loss (BCELoss)

Optimizer: Adam

⚙️ Training Details

Epochs: 100

Learning Rate: 0.001

Batching: Full batch (entire dataset used per epoch)

Device: Runs on GPU (CUDA) if available, else CPU

📈 Results

✅ Training Accuracy: ~99%
✅ Test Accuracy: ~96%

⚡ Your results may vary depending on random seed & hyperparameters.

🚀 How to Run

Clone this repository:

git clone https://github.com/Nayasha2003/breast-cancer-classification-pytorch.git
cd breast-cancer-classification-pytorch


Install dependencies:

pip install torch scikit-learn


Run the script:

python breast_cancer_classification.py

📂 Project Structure
📦 breast-cancer-classification-pytorch
 ┣ 📜 breast_cancer_classification.py   # Main training & evaluation script
 ┣ 📜 README.md                         # Project documentation

✨ Key Skills Demonstrated

🔹 PyTorch – Building and training deep learning models

🔹 Scikit-learn – Dataset handling, preprocessing & evaluation

🔹 Data Preprocessing – Feature scaling with StandardScaler

🔹 Model Deployment Readiness – GPU/CPU compatibility

🔹 Classification Metrics – Accuracy evaluation

📌 Future Improvements

✅ Add dropout layers to reduce overfitting

✅ Use cross-validation for more robust evaluation

✅ Experiment with deeper architectures and regularization

✅ Visualize training loss & accuracy curves

💡 Inspiration

This project is aimed at showcasing machine learning + deep learning skills for healthcare AI. It highlights end-to-end implementation of a classification pipeline with PyTorch.
