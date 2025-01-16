import streamlit as st
import requests
import json
import logging
from typing import Optional
from ultralytics import YOLO
from PIL import Image
import cv2

# Constants
BASE_API_URL = "http://127.0.0.1:7860"
FLOW_ID = "78287719-c459-48f1-84ed-db8473f0b34a"
ENDPOINT = ""  # You can set a specific endpoint name in the flow settings

TWEAKS = {
    "OpenAIModel-8BvkW": {},
    "Prompt-miH9T": {},
    "ChatInput-Qk60y": {},
    "ChatOutput-Fxdyc": {}
}

logging.basicConfig(level=logging.INFO)

# Function to run the Langflow API
def run_flow(message: str,
  endpoint: str = FLOW_ID,
  output_type: str = "chat",
  input_type: str = "chat",
  tweaks: Optional[dict] = None,
  api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    if tweaks:
        payload["tweaks"] = tweaks
    headers = None
    if api_key:
        headers = {"x-api-key": api_key}
    response = requests.post(api_url, json=payload, headers=headers)

    logging.info(f"Response Status Code: {response.status_code}")
    logging.info(f"Response Text: {response.text}")

    try:
        return response.json()
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON from the server response.")
        return {}

# Function to extract the assistant's message from the response
def extract_message(response: dict) -> str:
    try:
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."

# Load the YOLO model
TRAINED_MODEL_PATH = "best.pt"  # Replace with your actual model path
best_sign_language_model = YOLO(TRAINED_MODEL_PATH)

def main():
    st.title("Sign Language Interpreter Bot")
    st.write("Please allow for webcam access first before using this app.")

    run = st.checkbox("Run")
    FRAME_WINDOW = st.image([])
    gesture_display = st.empty()  # Placeholder for displaying detected gestures
    camera = cv2.VideoCapture(0)
    
    if not camera.isOpened():
        st.error("Failed to open camera.")
        return  # Exit gracefully

    retries = 3  # Retry capturing frame
    ret, frame = camera.read()
    while not ret and retries > 0:
        st.warning("Retrying to capture the frame...")
        ret, frame = camera.read()
        retries -= 1

    if not ret:
        st.error("Failed to capture frame after retries.")
        return  # Exit gracefully

    # Track detected gestures in the order they are detected
    detected_gestures_list = []

    while run:
        _, frame = camera.read()
        if frame is not None and frame.size > 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            st.error("No frame received or frame is empty")

        # Use YOLO to detect objects
        results = best_sign_language_model.predict(source=frame, stream=True)  # Use `frame` as input

        updated = False  # To check if the list was updated in this frame

        for result in results:
            for box in result.boxes:
                if box.conf >= 0.4: # Confidence score that can be adjusted to accept or reject gesture that is not confident enough
                    gesture_name = result.names[int(box.cls)]
                    if gesture_name not in detected_gestures_list:
                        detected_gestures_list.append(gesture_name)
                        updated = True

            frame = result.plot()  # This draws boxes on the frame

        FRAME_WINDOW.image(frame)

        if updated:
            gesture_message = ", ".join(detected_gestures_list)
            
            # Display detected gestures in the chat interface
            with st.chat_message("user"):
                st.write(f"Detected Gestures: {gesture_message}")

            # Prepare tweaks for Langflow API
            tweaks = TWEAKS.copy()  # Create a copy of the default tweaks
            tweaks["Prompt-miH9T"]["detected_word"] = gesture_message

            # Get chatbot response from Langflow API
            response = run_flow(gesture_message, tweaks=tweaks)
            chatbot_response = extract_message(response)

            # Display chatbot response in the chat interface
            with st.chat_message("assistant"):
                st.write(chatbot_response)

    # Release camera
    camera.release()

if __name__ == "__main__":
    main()