import os
import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image, ImageOps
from google import genai

# --- Load model and labels (files must be in the same folder as app.py) ---
saved_model = tf.saved_model.load("model.savedmodel")
infer = saved_model.signatures["serving_default"]

with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.split()[-1] for line in f if line.strip()]

# --- Gemini key comes from a secret, never from the code ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def predict(image, language):
    if image is None:
        return "Please upload a plant leaf image."
    language = (language or "English").strip()

    img = ImageOps.fit(image.convert("RGB"), (224, 224), Image.Resampling.LANCZOS)
    data = np.expand_dims((np.asarray(img).astype(np.float32) / 127.5) - 1, axis=0)

    output = infer(tf.convert_to_tensor(data))
    prediction = list(output.values())[0].numpy().flatten()
    index = int(np.argmax(prediction))
    class_name = class_names[index]
    confidence = float(prediction[index]) * 100

    healthy = "Healthy" in class_name
    plant = class_name.split("_")[0]
    status = "Healthy" if healthy else "Unhealthy"

    header = f"**Plant:** {plant}  \n**Status:** {status}  \n"
    if not healthy:
        header += f"**Disease:** {class_name}  \n"
    header += f"**Confidence:** {confidence:.2f}%\n\n---\n\n"

    if client is None:
        return header + "Gemini explanation unavailable: API key is not set."

    if healthy:
        task = ("Explain what it means for this plant to be healthy, give general tips "
                "to keep it healthy, and how to prevent common diseases.")
    else:
        task = ("Give a brief explanation of the disease, key symptoms, immediate actions, "
                "natural or low-cost remedies, when chemical treatment may be appropriate, "
                "and preventive measures. Do not claim the AI diagnosis is 100% certain; "
                "suggest confirming with an agricultural expert if symptoms are unclear.")

    prompt = (f"You are an expert agricultural plant pathologist. The classifier detected: "
              f"{class_name}. {task} Use simple, farmer-friendly language. "
              f"Answer ONLY in {language}.")
    try:
        resp = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        return header + resp.text
    except Exception as e:
        return header + f"Gemini explanation unavailable: `{e}`"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="🌿 Upload a Plant Leaf Image"),
        gr.Textbox(label="🌐 Preferred Language", value="English",
                   placeholder="English, Tamil, Hindi..."),
    ],
    outputs=gr.Markdown(label="🤖 AI-Powered Disease Explanation"),
    title="🌱 AI Plant Disease Detection",
    description="Upload a leaf photo to detect the disease and get farmer-friendly guidance.",
)

if __name__ == "__main__":
    demo.launch()
