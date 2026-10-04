import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from google import genai

st.set_page_config(page_title="AI Plant Disease Detection", page_icon="🌱")


@st.cache_resource
def load_model():
    saved_model = tf.saved_model.load("model.savedmodel")
    infer = saved_model.signatures["serving_default"]
    with open("labels.txt", "r", encoding="utf-8") as f:
        class_names = [line.split()[-1] for line in f if line.strip()]
    return infer, class_names


def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None
    return genai.Client(api_key=key) if key else None


infer, class_names = load_model()
client = get_client()

st.title("🌱 AI Plant Disease Detection")
st.write("Upload a leaf photo to detect the disease and get farmer-friendly guidance.")

uploaded = st.file_uploader("🌿 Upload a plant leaf image", type=["jpg", "jpeg", "png"])
language = st.text_input("🌐 Preferred language", value="English")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Analyze"):
        img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        data = np.expand_dims((np.asarray(img).astype(np.float32) / 127.5) - 1, axis=0)

        output = infer(tf.convert_to_tensor(data))
        prediction = list(output.values())[0].numpy().flatten()
        index = int(np.argmax(prediction))
        class_name = class_names[index]
        confidence = float(prediction[index]) * 100

        healthy = "Healthy" in class_name
        plant = class_name.split("_")[0]

        st.markdown(f"**Plant:** {plant}  \n**Status:** {'Healthy' if healthy else 'Unhealthy'}")
        if not healthy:
            st.markdown(f"**Disease:** {class_name}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.divider()

        if client is None:
            st.warning("Gemini explanation unavailable: API key is not set.")
        else:
            if healthy:
                task = ("Explain what it means for this plant to be healthy, give general "
                        "tips to keep it healthy, and how to prevent common diseases.")
            else:
                task = ("Give a brief explanation of the disease, key symptoms, immediate "
                        "actions, natural or low-cost remedies, when chemical treatment may "
                        "be appropriate, and preventive measures. Do not claim the AI "
                        "diagnosis is 100% certain; suggest confirming with an agricultural "
                        "expert if symptoms are unclear.")
            prompt = (f"You are an expert agricultural plant pathologist. The classifier "
                      f"detected: {class_name}. {task} Use simple, farmer-friendly language. "
                      f"Answer ONLY in {language.strip() or 'English'}.")
            with st.spinner("Generating explanation..."):
                try:
                    resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                    st.markdown(resp.text)
                except Exception as e:
                    st.error(f"Gemini explanation unavailable: {e}")
