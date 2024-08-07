import streamlit as st
import torch
import torchvision.transforms as T
from transformers import DetrImageProcessor, DetrForObjectDetection, CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests

# Here are trying to Load a models
detr_model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# It is Helping function
def detect_objects(image):
    processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')
    inputs = processor(images=image, return_tensors="pt")
    outputs = detr_model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    return results

# Here generating a caption of the image 
def generate_caption(image):
    text = "a photography of"
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    return text, probs

# Here generating a blip caption which is effectively leverages noisy web data through a bootstrapping mechanism, 
# where a captioner generates synthetic captions filtered by a noise removal process.  
def generate_blip_caption(image):
    inputs = blip_processor(image, return_tensors="pt")
    outputs = blip_model.generate(**inputs)
    caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Streamlit app
st.title("Object Detection and Captioning")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Object detection
    results = detect_objects(image)
    st.write("Detected Objects:")
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        st.write(f"Label: {label}, Score: {score:.4f}, Box: {box}")

    # CLIP caption generation
    text, probs = generate_caption(image)
    st.write(f"CLIP Caption: {text}, Probability: {probs}")

    # BLIP caption generation
    blip_caption = generate_blip_caption(image)
    st.write(f"BLIP Caption: {blip_caption}")
