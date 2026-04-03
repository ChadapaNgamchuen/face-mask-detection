import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import streamlit as st

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["with_mask", "without_mask"]


@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("mask_model.pth", map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

model = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


st.title("Face Mask Detection")
st.caption("Upload รูปภาพ แล้วโมเดลจะบอกว่าใส่หน้ากากหรือเปล่า")

uploaded = st.file_uploader("เลือกรูปภาพ", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="รูปที่อัปโหลด", use_container_width=True)

    
    with torch.no_grad():
        tensor = transform(image).unsqueeze(0).to(DEVICE)
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]

    with_mask_score    = probs[0].item()
    without_mask_score = probs[1].item()
    predicted          = CLASSES[probs.argmax().item()]

    st.divider()

   
    if predicted == "with_mask":
        st.success("With Mask")
    else:
        st.error("Without Mask")

    st.write("### Confidence Score")
    st.progress(with_mask_score,    text=f"With Mask:    {with_mask_score:.2%}")
    st.progress(without_mask_score, text=f"Without Mask: {without_mask_score:.2%}")