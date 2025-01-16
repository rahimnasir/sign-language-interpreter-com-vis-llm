#%%
import streamlit as st
import requests
import json
import logging
from typing import Optional
from ultralytics import YOLO
from ultralytics import YOLOWorld
from PIL import Image

#%% Constant
BASE_API_URL = "http://127.0.0.1:7860"
FLOW_ID = "78287719-c459-48f1-84ed-db8473f0b34a"
ENDPOINT = "" # You can set a specific endpoint name in the flow settings

# You can tweak the flow by adding a tweaks dictionary
# e.g {"OpenAI-XXXXX": {"model_name": "gpt-4"}}
TWEAKS = {
  "OpenAIModel-8BvkW": {},
  "Prompt-miH9T": {},
  "ChatInput-Qk60y": {},
  "ChatOutput-Fxdyc": {}
}

# Initialize logging
logging.basicConfig(level=logging.INFO)

def run_flow(message: str,
  endpoint: str = FLOW_ID,
  output_type: str = "chat",
  input_type: str = "chat",
  tweaks: Optional[dict] = None,
  api_key: Optional[str] = None) -> dict:
    """
    Run a flow with a given message and optional tweaks.

    :param message: The message to send to the flow
    :param endpoint: The ID or the endpoint name of the flow
    :param tweaks: Optional tweaks to customize the flow
    :return: The JSON response from the flow
    """
    api_url = f"{BASE_API_URL}/api/v1/run/{endpoint}"

    payload = {
        "input_value": message,
        "output_type": output_type,
        "input_type": input_type,
    }
    headers = None
    if tweaks:
        payload["tweaks"] = tweaks
    if api_key:
        headers = {"x-api-key": api_key}
    response = requests.post(api_url, json=payload, headers=headers)

    # Log the response for debugging 

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
        # Extract the response message
        return response['outputs'][0]['outputs'][0]['results']['message']['text']
    except (KeyError, IndexError):
        logging.error("No valid message found in response.")
        return "No valid message found in response."

def main():
    st.title("Sign Language Interpreter Bot")