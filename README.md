# OmniGuardAI - Text and Audio Cyberbullying Detection

## Overview
OmniGuardAI is a web-based application designed to detect cyberbullying in both text and audio inputs. The application utilizes pre-trained Natural Language Processing (NLP) models to classify text sentiment and detect toxicity, along with an automatic speech recognition (ASR) model to transcribe audio files for analysis.

## Features
- *Text-Based Detection*: Classifies text as positive or negative and flags cyberbullying if toxicity is detected.
- *Audio-Based Detection*: Converts speech to text and performs the same classification and toxicity analysis.
- *Justification for Labels*: Uses Gemini AI to provide reasoning for classifications.

## Technologies Used
- *Programming Language*: Python
- *Framework*: Streamlit
- *Libraries*:
  - numpy: For numerical operations.
  - pydub: For audio processing.
  - streamlit: For building the web interface.
  - transformers: For NLP models (sentiment analysis, toxicity classification, and Automatic Speech Recognition).
  - google.genai: For AI-generated justifications.
  - io.BytesIO: For handling audio file input.

## Installation
### Prerequisites
Ensure that you have Python installed (preferably version 3.8 or later). Install the required dependencies using the following command:
bash
pip install numpy pydub streamlit transformers google-generativeai


## How to Run
1. Clone the repository or save the script as omniguardai.py.
2. Run the following command to start the Streamlit application:
bash
streamlit run omniguardai.py

3. Access the application in your browser at http://localhost:8501/.

## Usage
1. *Text Detection*:
   - Enter the text into the provided text area.
   - Click "Check Text" to analyze.
   - The result and justification will be displayed.

2. *Audio Detection*:
   - Upload a .wav file.
   - Click "Check Audio" to analyze.
   - The transcribed text, classification result, and justification will be displayed.

## Model Details
- *Automatic Speech Recognition Model*: facebook/wav2vec2-large-960h
- *Sentiment Analysis Model*: distilbert-base-uncased-finetuned-sst-2-english
   This model is used to classify whether the text is positive or negative. If text is negative, the text is sent Toxicity Classification Model
- *Toxicity Classification Model*: unitary/unbiased-toxic-roberta
   This model classifies whether the text consists of cyberbullying text or not

## API Key Configuration
To use Gemini AI for label verification, update the API key in the script:
python
API_key = "YOUR_API_KEY"

Make sure to replace it with a valid key.

## Error Handling
- If an invalid or corrupted audio file is uploaded, an error message will be displayed.
- If the API fails to return a justification, the application will notify the user.
