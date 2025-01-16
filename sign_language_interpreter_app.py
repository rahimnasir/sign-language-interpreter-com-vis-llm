import streamlit as st
import requests
import json
import logging
from typing import Optional
from ultralytics import YOLO
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

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
TRAINED_MODEL_PATH = "best.pt"  # Path to your YOLO model in the repository
best_sign_language_model = YOLO(TRAINED_MODEL_PATH)

class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.detected_gestures_list = []

    def recv(self, frame: av.VideoFrame):
        frame = frame.to_ndarray(format="bgr24")
        results = best_sign_language_model.predict(source=frame, stream=True)
        updated = False

        for result in results:
            for box in result.boxes:
                if box.conf >= 0.4:  # Adjust confidence threshold
                    gesture_name = result.names[int(box.cls)]
                    if gesture_name not in self.detected_gestures_list:
                        self.detected_gestures_list.append(gesture_name)
                        updated = True

            frame = result.plot()

        # Display detected gestures
        if updated:
            gesture_message = ", ".join(self.detected_gestures_list)
            tweaks = TWEAKS.copy()
            tweaks["Prompt-miH9T"]["detected_word"] = gesture_message
            response = run_flow(gesture_message, tweaks=tweaks)
            chatbot_response = extract_message(response)
            st.session_state.gestures = gesture_message
            st.session_state.response = chatbot_response

        return av.VideoFrame.from_ndarray(frame, format="bgr24")

def main():
    st.title("Sign Language Interpreter Bot")
    st.write("This app uses a YOLO model to detect sign language gestures.")

    if "gestures" not in st.session_state:
        st.session_state.gestures = ""
    if "response" not in st.session_state:
        st.session_state.response = ""

    webrtc_streamer(
        key="sign-language",
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    st.write("### Detected Gestures")
    st.write(st.session_state.gestures)

    st.write("### Chatbot Response")
    st.write(st.session_state.response)

if __name__ == "__main__":
    main()