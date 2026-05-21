# Digital Photo Composition Assessment System

This project is a web-based system for assessing digital photo composition using transfer learning with MobileNetV2. The system predicts whether a photo has good composition or needs improvement, then provides visual feature-based recommendations to help users understand and improve their photo composition.

## Project Overview

Photo composition plays an important role in photography because it affects how visual elements are arranged and perceived by viewers. However, evaluating photo composition can be subjective, especially for beginner photographers.

This project applies a data science and computer vision pipeline to build a digital photo composition assessment system. The model is trained using the CADB (Image Composition Assessment Dataset), and the final system is implemented as a Streamlit web application.

The system can:

- Upload and analyze a photo
- Predict whether the photo composition is good or needs improvement
- Display prediction confidence
- Extract visual features from the image
- Provide visual composition recommendations
- Generate a downloadable analysis report

## Features

- Photo upload interface
- MobileNetV2 transfer learning model
- Ready-to-run trained model included
- Binary classification:
  - `baik` / good
  - `perlu_perbaikan` / needs improvement
- Visual feature extraction:
  - Brightness
  - Contrast
  - Sharpness
  - Edge density
  - Colorfulness
  - Rule of thirds score
  - Symmetry score
  - Aspect ratio
- Rule-based visual recommendation system
- Streamlit web interface
- Downloadable photo analysis report

## Quick Start

This repository already includes the trained MobileNetV2 model, so the web application can be run directly without retraining.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/photo-composition-assessment.git
cd photo-composition-assessment
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS or Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app_mobilenet.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Dataset

This project uses the CADB dataset:

**CADB - Image Composition Assessment Dataset**  
GitHub: https://github.com/bcmi/Image-Composition-Assessment-Dataset-CADB

The dataset contains 9,497 images with composition scores rated by human annotators. The original composition scores are converted into binary labels for classification.

| Mean Score | Label |
|---|---|
| `< 3.0` | `perlu_perbaikan` |
| `>= 3.0` | `baik` |

The dataset is not included in this repository because of its large file size. Please download it manually from the official dataset source if you want to retrain the model.

## Model

The main model uses MobileNetV2 with transfer learning. MobileNetV2 is used as a feature extractor, followed by additional classification layers for binary classification.

The trained model files are included in this repository:

```text
model/
├── mobilenet_composition_model.h5
└── mobilenet_label_classes.npy
```

These files are required to run the Streamlit application directly.

The project also includes a baseline model using manually extracted visual features and Random Forest for comparison.

### Model Comparison

| Model | Accuracy |
|---|---:|
| Random Forest with manual visual features | 67.79% |
| MobileNetV2 Transfer Learning | 74.73% |

The MobileNetV2 model achieved better performance and is used as the main model in the web application.

## Project Structure

```text
photo-composition-assessment/
│
├── app_mobilenet.py
├── train_mobilenet.py
├── train_model_binary.py
├── extract_features.py
├── extract_features_binary.py
├── prepare_labels.py
├── prepare_labels_binary.py
├── check_dataset.py
├── requirements.txt
├── README.md
│
├── model/
│   ├── mobilenet_composition_model.h5
│   └── mobilenet_label_classes.npy
│
└── assets/
    ├── app_main.png
    └── app_recommendation.png
```

The `dataset/` folder is not included in the repository because the dataset is large. The trained MobileNetV2 model files are included so users can run the app immediately.

## Requirements

The main dependencies are:

```txt
streamlit
tensorflow
numpy
opencv-python
pillow
scikit-learn
pandas
matplotlib
joblib
tqdm
```

## Running Without Training

To run the application without training, make sure the following files exist:

```text
model/
├── mobilenet_composition_model.h5
└── mobilenet_label_classes.npy
```

Then run:

```bash
streamlit run app_mobilenet.py
```

The application will load the trained MobileNetV2 model automatically.

## Dataset Preparation for Training

This step is only required if you want to retrain the model.

After downloading the CADB dataset, place it inside the project folder using this structure:

```text
dataset/
└── CADB_Dataset/
    ├── composition_scores.json
    ├── composition_attributes.json
    ├── composition_elements.json
    ├── scene_categories.json
    └── images/
```

Check whether the dataset is readable:

```bash
python check_dataset.py
```

Prepare binary labels:

```bash
python prepare_labels_binary.py
```

## Training

This step is optional because the trained model is already included.

To train the MobileNetV2 transfer learning model:

```bash
python train_mobilenet.py
```

The trained model will be saved in the `model/` directory:

```text
model/
├── mobilenet_composition_model.h5
└── mobilenet_label_classes.npy
```

The dataset is split into:

- 70% training data
- 15% validation data
- 15% testing data

In this project, the MobileNetV2 model was trained using 9,497 images from CADB, with:

| Data Split | Number of Images |
|---|---:|
| Training | 6,647 |
| Validation | 1,425 |
| Testing | 1,425 |

## How the System Works

The system follows this workflow:

1. The user uploads a photo.
2. The image is resized and preprocessed.
3. The MobileNetV2 model predicts the composition class.
4. Visual features are extracted using OpenCV.
5. The system displays:
   - Prediction result
   - Confidence score
   - Visual feature values
   - Good visual aspects
   - Improvement recommendations
6. The user can download the analysis result as a text file.

## Visual Features

The system extracts several supporting visual features:

| Feature | Description |
|---|---|
| Brightness | Average intensity of the image |
| Contrast | Standard deviation of grayscale intensity |
| Sharpness | Laplacian variance for blur estimation |
| Edge Density | Ratio of edge pixels detected using Canny edge detection |
| Colorfulness | Color variation based on RGB channel differences |
| Rule of Thirds Score | Approximation of edge distribution near rule-of-thirds lines |
| Symmetry Score | Horizontal visual balance estimation |
| Aspect Ratio | Width-to-height ratio of the image |

These features are used to support the recommendation system, not as the main prediction model.

## Recommendation System

The recommendation system is rule-based. It uses extracted visual features to generate feedback related to:

- Focus and sharpness
- Exposure and brightness
- Contrast
- Color intensity
- Rule of thirds
- Visual balance
- Background complexity
- Foreground and background distraction
- Image orientation and aspect ratio

The final prediction is produced by the MobileNetV2 model, while the recommendations are generated using visual feature rules.

## Example Output

```text
Prediction: baik
Confidence: 81.23%
Confidence Level: Sedang

Good Aspects:
- The image has strong sharpness and clear visual details.
- Some visual elements are close to the rule-of-thirds area.
- The image has strong contrast between object and background.

Recommendations:
- The left and right visual balance can still be improved through slight cropping.
```

## Results

The MobileNetV2 transfer learning model achieved the following result on the test set:

```text
Accuracy: 74.73%
```

Classification report:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| baik | 0.61 | 0.69 | 0.65 |
| perlu_perbaikan | 0.83 | 0.78 | 0.80 |

This result shows that MobileNetV2 performs better than the baseline Random Forest model using manual visual features.

## Limitations

This system has several limitations:

- Photo composition is subjective.
- The system only classifies photos into two categories.
- Recommendations are generated using rule-based visual features.
- The model does not yet detect photography genres such as portrait, landscape, architecture, or street photography.
- The system does not yet detect specific composition types such as leading lines, framing, negative space, or centered composition.

## Future Development

Future improvements may include:

- Photography genre classification
- Composition type detection
- Multi-class composition quality prediction
- More advanced models such as EfficientNet, ResNet, ConvNeXt, or Vision Transformer
- Better explainability using Grad-CAM or saliency maps
- More advanced natural language recommendation generation
- Web deployment to cloud platforms

## Technologies Used

- Python
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Pillow

## Author

M. Hisyam Al Firdaus

## License

This project is created for academic and educational purposes.