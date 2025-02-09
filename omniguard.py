import numpy as np
from pydub import AudioSegment
import streamlit as st
from transformers import pipeline
from google import genai
from io import BytesIO

# Load models once to avoid reloading them multiple times
@st.cache_resource
def load_models():
    """Load the pretrained sentiment and toxicity classification models."""
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    toxicity_model = pipeline("text-classification", model="unitary/unbiased-toxic-roberta", return_all_scores=True)
    asr_pipeline = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h")
    return sentiment_model, toxicity_model, asr_pipeline

sentiment_model, toxicity_model, asr_pipeline = load_models()

# Convert audio file to numpy array
def audio_to_numpy(audio_file):
    # Load audio file using pydub
    audio = AudioSegment.from_wav(audio_file)

    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    
    # Set sample width and frame rate (important for speech-to-text)
    audio = audio.set_sample_width(2).set_frame_rate(16000)

    # Convert to raw audio data
    audio_data = np.array(audio.get_array_of_samples())
    return audio_data

def classify_text(text):
    """Classifies text into Positive/Negative and detects Cyberbullying if applicable."""
    sentiment_result = sentiment_model(text)[0]
    label = sentiment_result["label"]

    if label == "POSITIVE":
        return {"sentiment": "Positive", "final_label": "None"}

    # Get toxicity scores
    toxicity_results = toxicity_model(text)
    toxicity_scores = {entry['label']: entry['score'] for entry in toxicity_results[0]}

    relevant_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult"]
    toxicity_threshold = 0.4

    # Check if toxicity threshold is exceeded
    if any(toxicity_scores.get(label, 0) > toxicity_threshold for label in relevant_labels):
        return {"sentiment": "Negative", "final_label": "Cyberbullying"}

    return {"sentiment": "Negative", "final_label": "None"}

def verify_label(text, label):
    """Uses Gemini AI to provide a justification for classification."""
    API_key = "YOUR_API_KEY"  # Ensure that the API key is set up correctly in the environment
    client = genai.Client(api_key=API_key)  # Replace with your actual API key

    content = (
        f"""You are an AI tasked with verifying the correctness of a label assigned to a given text. 
        Text: "{text}"
        Label: "{label}"
        
        + """Your task is to:
        - Analyze the text's meaning, tone, and intent (e.g., sarcasm, humor, aggression, neutral).
        - Provide a reasoned two-sentence explanation based on linguistic cues, intent, and context.
        - Suggest a more appropriate label if necessary. Do NOT mention if the label is correct; just justify it."""
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=content
    )
    return response.text

def main():
    st.title("OmniGuardAI - Text and Audio Cyberbullying Detection")

    # Creating buttons
    if st.button("Check Text"):
        st.session_state["option"] = "text"
    if st.button("Check Audio"):
        st.session_state["option"] = "audio"

    # Handling Text Input
    if st.session_state.get("option") == "text":
        st.header("Cyberbullying Text Detection")
        text_input = st.text_area("Enter Text", "")

        if text_input:
            result = classify_text(text_input)
            justification = verify_label(text_input, result["final_label"])
            st.write(f"*Result:* {result['final_label']}")
            st.write(f"*Justification:* {justification}")

    # Handling Audio Input
    elif st.session_state.get("option") == "audio":
        st.header("Cyberbullying Audio Detection")
        audio_file = st.file_uploader("Upload Audio (.wav format)", type=["wav"])

        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")

            # Debugging: Check the type of the uploaded file
            #st.write(f"Audio file type: {type(audio_file)}")

            # Convert audio file to numpy array
            try:
                audio_data = audio_to_numpy(audio_file)

                # Debugging: Print out the shape of the numpy array
                #st.write(f"Audio data shape: {audio_data.shape}")

                # Convert numpy array to correct format for ASR pipeline
                transcription = asr_pipeline(audio_data)["text"]
                st.write(f"*Transcribed Text:* {transcription}")

                # Classify transcribed text
                result = classify_text(transcription)
                justification = verify_label(transcription, result["final_label"])
                st.write(f"Result: {result['final_label']}")
                st.write(f"Justification: {justification}")

            except Exception as e:
                st.write(f"Error in processing audio file: {str(e)}")

if __name__ == "__main__":
    main()
