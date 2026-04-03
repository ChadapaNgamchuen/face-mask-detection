# Face Mask Detection
To build a binary image classifier Mask vs. No Mask using a pre-trained ResNet18 model

## Model
- Architecture: ResNet18 (Transfer Learning)
- Dataset: [7,553 picture dataset (Kaggle)](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- Best Val Accuracy: 99.01% (Epoch 10)

## Installation
pip install -r requirements.txt

## Train
python train.py

## Run App
streamlit run app.py

## Results
![Training Results](training_results.png)

## Output
| Without Mask | With Mask |
|---|---|
| <img width="407" height="686" alt="without_mask" src="https://github.com/user-attachments/assets/851123ed-8092-447f-abf9-de755c99be0c" /> | <img width="401" height="665" alt="with_mask" src="https://github.com/user-attachments/assets/376d6f11-6823-489a-82cf-9a59c42a0fb3" /> |
