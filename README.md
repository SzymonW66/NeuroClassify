# Brain Tumor Classification

A web application that demonstrates brain image classification using two machine learning models: EfficientNet and VGG. After an image is uploaded, the application displays the predicted class from each model.

## Classification categories

- Glioma Tumor
- No Tumor
- Meningioma Tumor
- Pituitary Tumor

## Requirements

- Python
- Git LFS — the model files are stored using Git LFS

After cloning the repository, make sure the LFS models are downloaded:

```powershell
git lfs install
git lfs pull
```

## Running on Windows

From the project directory, create a virtual environment, install the dependencies, and run the application:

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install Flask tensorflow pillow numpy
.venv\Scripts\python app.py
```

Open the address shown in the terminal (by default, `http://127.0.0.1:5000`) and upload a brain image. The application limits uploads to 16 MB. Each new upload deletes the previous files from the `uploads` directory.

## Project structure

- `app.py` — Flask application, image preprocessing, and predictions
- `model1/effnet.keras` — EfficientNet model
- `model2/vgg.h5` — VGG model
- `templates/index.html` — application interface
- `static/images/` — image assets
- `uploads/` — uploaded images

## About the project

This is a **group project submitted as part of a university course**. It is intended for educational and demonstration purposes only. Model predictions are not a medical diagnosis or medical advice.

This README was created with the help of AI.
