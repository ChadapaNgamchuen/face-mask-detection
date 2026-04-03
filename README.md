# Face Mask Detection
จำแนกรูปภาพว่าใส่หน้ากากหรือไม่ ด้วย PyTorch + ResNet18

## Model
- Architecture: ResNet18 (Transfer Learning)
- Dataset: [7,553 รูป (Kaggle)](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)
- Best Val Accuracy: 99.01% (Epoch 10)

## Installation
pip install -r requirements.txt

## Train
python train.py

## Run App
streamlit run app.py