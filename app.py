import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the BLIP model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Function to generate captions
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# Streamlit App
def app():
    st.title("Image Captioning with BLIP")
    st.markdown("Upload an image, and the model will generate a caption!")

    # Upload image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Display image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Generate and display caption
        if st.button("Generate Caption"):
            caption = generate_caption(image)
            st.write("Caption:", caption)

if __name__ == "__main__":
    app()
