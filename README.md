# MNIST Classification & Web Interface

This repository contains a complete machine learning pipeline for classifying handwritten digits from the MNIST dataset. It includes PyTorch model training scripts, a FastAPI inference server, a Streamlit interactive drawing canvas frontend, and configuration files for Docker and GitHub/Cloud deployment.

---

## 🚀 Features

- **Interactive Streamlit Frontend** (`app.py`):
  - Canvas interface for drawing digits directly in the browser.
  - Live prediction displays with predicted digit, confidence scores, and a probability chart.
  - Integrated **MNIST-compliant preprocessing** (digit centering) to guarantee high model accuracy.
- **FastAPI Inference Server** (`main.py`):
  - High-performance API endpoint (`/predict`) serving model inferences.
  - Returns predicted class, model confidence, and full probability distributions.
- **Robust Digit Centering** (`center_digit`):
  - Uses bounding-box cropping, aspect-ratio preserving scaling, and center-of-mass calculation via `scipy.ndimage` to replicate the standard MNIST dataset preprocessing.
  - Elevates prediction confidence from ~0.48 to ~0.998 on off-center drawings.
- **Docker Support** (`Dockerfile`):
  - Multi-platform deployable container configured with Python 3.12.
  - Utilizes shell form for `CMD` to support dynamic environment ports (e.g., on Render).
- **Environment Agnostic Configuration**:
  - Dynamically switches backend targets via the `BACKEND_URL` environment variable.
- **Dev Container Integration** (`.devcontainer/devcontainer.json`):
  - Out-of-the-box support for VS Code Dev Containers/GitHub Codespaces to launch backend/frontend instantly.

---

## 📁 Project Structure

```text
├── .devcontainer/
│   └── devcontainer.json     # Configuration for VS Code Dev Containers / Codespaces
├── model/
│   └── chkpoint.pth          # Saved trained CNN model weights
├── .dockerignore             # Excludes source/metadata files from Docker context
├── .gitignore                # Git ignore rules (datasets, caches, environment configs)
├── Dockerfile                # Build instructions for containerizing the FastAPI app
├── app.py                    # Streamlit frontend with drawing canvas & preprocessing logic
├── main.py                   # FastAPI application serving predictions
├── mnist_cnn.py              # CNN model architecture & training script
├── mnist_from_scratch.py     # Simple feedforward neural network training script
├── requirements.txt          # Frontend dependencies (Streamlit Cloud convention)
├── render_requirements.txt   # Backend dependencies (Render platform convention)
├── test_file.py              # Helper to extract and test raw MNIST image arrays
└── README.md                 # Project documentation
```

---

## 🛠️ Setup & Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Adityaa10101/MNIST.git
cd MNIST
```

### 2. Configure Virtual Environment & Install Dependencies
Create a virtual environment and install the required modules:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install requirements for local development
pip install -r requirements.txt -r render_requirements.txt
```

---

## 🏋️ Training the Models

### CNN Model (Recommended)
Trains a Convolutional Neural Network (CNN) in PyTorch and updates the checkpoint:
```bash
python mnist_cnn.py
```
This script downloads the MNIST dataset to `./root`, trains for 5 epochs, logs testing accuracy, and outputs the model weights to `./model/chkpoint.pth`.

### Feedforward Network
Trains a standard three-layer fully-connected Neural Network from scratch:
```bash
python mnist_from_scratch.py
```
Trains for 15 epochs and prints training/test losses and accuracy.

---

## 🖥️ Running Locally

### 1. Start the FastAPI Backend
Start the server locally using `uvicorn`:
```bash
uvicorn main:app --reload
```
The API documentation will be available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### 2. Start the Streamlit Frontend
In a new terminal window, configure the backend URL and run Streamlit:
```bash
# Windows
set BACKEND_URL=http://127.0.0.1:8000
streamlit run app.py

# macOS/Linux
BACKEND_URL=http://127.0.0.1:8000 streamlit run app.py
```
The Streamlit app will load automatically in your browser at `http://localhost:8501`.

---

## 🧠 MNIST Digit Preprocessing & Centering

Raw canvas sketches are often off-center or drawn with varying scale, resulting in poor prediction accuracy (e.g., confidence of ~0.48 or incorrect classification). To match the MNIST dataset distribution, the `center_digit()` preprocessing pipeline in [app.py](file:///c:/Users/Lenovo/Desktop/ML/app.py) performs the following operations:
1. **Bounding Box Crop**: Isolates the drawn digit by cropping all empty space (pixels <= 50).
2. **Aspect-Preserving Resize**: Scales the cropped image so the longer side is exactly 20 pixels.
3. **Canvas Padding**: Centers the resized digit on a blank 28x28 canvas (providing a 4-pixel border).
4. **Center-of-Mass Alignment**: Shifts the digit mathematically using `scipy.ndimage.center_of_mass` and `scipy.ndimage.shift` so its center of mass aligns exactly with the center of the 28x28 canvas (14, 14).

**Result**: Centering elevates prediction confidence on the same drawings from ~0.48 to ~0.998.

---

## 🐳 Docker Deployment

The FastAPI application can be packaged into a lightweight Docker image.

### 1. Build the Docker Image
```bash
docker build -t mnist-server .
```

### 2. Run the Container
```bash
docker run -p 8000:8000 -e PORT=8000 mnist-server
```

---

## 🌐 Cloud Deployment

The application is deployed end-to-end:

### Backend (Render)
- **Service**: Web Service (Docker-based)
- **URL**: [https://draw-a-digit-01d8.onrender.com](https://draw-a-digit-01d8.onrender.com)
- **Configuration Note**: The `Dockerfile` uses the shell form for `CMD` (`CMD uvicorn main:app --host 0.0.0.0 --port $PORT`) so that Render's dynamic port assignment variable `$PORT` is expanded correctly.
- **Dependencies**: Leverages [render_requirements.txt](file:///c:/Users/Lenovo/Desktop/ML/render_requirements.txt) to include PyTorch (CPU-only variant) to reduce image size and build times.

### Frontend (Streamlit Community Cloud)
- **URL**: [https://draw-a-digit.streamlit.app](https://draw-a-digit.streamlit.app)
- **Deployment Details**: 
  - To prevent dependency conflicts and keep build images slim, frontend-specific packages (e.g., `streamlit-drawable-canvas`, `scipy`) are isolated in [requirements.txt](file:///c:/Users/Lenovo/Desktop/ML/requirements.txt) which Streamlit Cloud automatically picks up by default.
  - Python version pinned to **3.12** inside the deployment dashboard to resolve segfaults caused by immature NumPy/SciPy wheel builds in newer, pre-release python releases (like Python 3.14).

---

## ⚠️ Known Limitations
- **Single-Digit Centering**: The current centering implementation assumes a single digit is drawn. Multi-digit inputs or disjoint strokes will result in calculation shifts that break centering and lead to incorrect predictions.
