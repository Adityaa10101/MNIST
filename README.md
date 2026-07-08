# MNIST Classification & FastAPI Inference Server

This repository contains a complete, lightweight machine learning pipeline for classifying handwritten digits from the MNIST dataset. It includes model definitions in PyTorch, training scripts (both a simple Feedforward Network and a Convolutional Neural Network), model checkpointing, and a FastAPI inference server to serve predictions.

## 🚀 Features

- **Training Scripts**:
  - `mnist_from_scratch.py`: Trains a classic fully-connected Neural Network (multilayer perceptron) from scratch.
  - `mnist_cnn.py`: Trains a Convolutional Neural Network (CNN) with PyTorch and saves checkpoints to the `model/` directory.
- **FastAPI Inference Server** (`main.py`):
  - Exposes an API endpoint (`/predict`) to predict handwritten digits using the trained CNN model checkpoint.
- **Docker Support**:
  - `Dockerfile` and `.dockerignore` files for building a lightweight containerized version of the FastAPI server.
- **Testing Utility** (`test_file.py`):
  - Utility to load a sample image from the MNIST test set and generate a JSON payload suitable for testing the API server.

---

## 📁 Project Structure

```text
├── model/
│   └── chkpoint.pth          # Saved trained model weights
├── .dockerignore             # Files to ignore in the Docker context
├── .gitignore                # Git ignore file (excludes datasets/caches)
├── Dockerfile                # Instructions to containerize the FastAPI application
├── main.py                   # FastAPI application for inference
├── mnist_cnn.py              # CNN model training script
├── mnist_from_scratch.py     # Feedforward neural network training script
├── requirements.txt          # Python dependencies
├── test_file.py              # Helper to extract test sample for API testing
└── README.md                 # Project documentation
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Adityaa10101/MNIST.git
cd MNIST
```

### 2. Install Dependencies
Make sure you have Python installed, then install the required packages:
```bash
pip install torch torchvision fastapi uvicorn pydantic
```

---

## 🏋️ Training the Models

### CNN Model (Recommended)
To train the CNN model and update the checkpoint:
```bash
python mnist_cnn.py
```
This script downloads the MNIST dataset to `./root`, runs training for 5 epochs, prints the test accuracy, and saves the trained state dictionary to `./model/chkpoint.pth`.

### Feedforward Network
To train the fully-connected neural network:
```bash
python mnist_from_scratch.py
```
This script trains a three-layer network for 15 epochs and prints the training/test losses and accuracy.

---

## 🖥️ Running the FastAPI Inference Server

Start the API server locally using `uvicorn`:
```bash
uvicorn main:app --reload
```

The server will be running on `http://127.0.0.1:8000`. You can visit the interactive API documentation at:
- Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- ReDoc: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🐳 Docker Deployment

You can also package and run the FastAPI server inside a Docker container.

### 1. Build the Docker Image
From the repository root, build the Docker image:
```bash
docker build -t mnist-server .
```

### 2. Run the Container
Run the container, mapping port `8000` to your host machine:
```bash
docker run -p 8000:8000 mnist-server
```
Once started, the API will be available at `http://localhost:8000`.

---

## 🧪 Testing the API

To send a prediction request to the FastAPI server:

1. **Generate test data**:
   Run `test_file.py` to extract the first image from the MNIST test set and output its pixel values as a JSON payload:
   ```bash
   python test_file.py
   ```

2. **Send request via curl**:
   Use the output from `test_file.py` in a POST request:
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict" \
        -H "Content-Type: application/json" \
        -d "{\"pixel_values\": [<784 comma-separated float values>]}"
   ```
   **Expected Response:**
   ```json
   {
     "Predicted_Class": 7,
     "Prob": 0.9984
   }
   ```
