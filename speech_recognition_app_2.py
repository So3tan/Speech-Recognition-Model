
import streamlit as st
import speech_recognition as sr
import whisper
import tempfile
import os

# Initialize Whisper model globally to avoid reloading
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def transcribe_with_google(audio_data, language):
    recognizer = sr.Recognizer()
    try:
        return recognizer.recognize_google(audio_data, language=language)
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError as e:
        return f"Google API error: {e}"

def transcribe_with_whisper(audio_data):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(audio_data.get_wav_data())
        temp_filename = temp_file.name

    model = load_whisper_model()
    result = model.transcribe(temp_filename)
    os.remove(temp_filename)
    return result["text"]

def transcribe_speech(api_choice, language):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Adjusting for background noise...")
        recognizer.adjust_for_ambient_noise(source)
        st.info("Listening...")
        audio_data = recognizer.listen(source, phrase_time_limit=10)

    if api_choice == "Google":
        return transcribe_with_google(audio_data, language)
    elif api_choice == "Whisper":
        return transcribe_with_whisper(audio_data)
    else:
        return "Invalid API selected."

def main():
    st.title("🎤 Speech Recognition App (Google & Whisper)")
    st.markdown("Select your speech recognition API, language, and then start speaking!")

    # API selection
    api_choice = st.selectbox("Select API", ["Google", "Whisper"])

    # Language selection (only for Google)
    language = st.selectbox("Select language", [
        ("English (US)", "en-US"),
        ("Spanish", "es-ES"),
        ("French", "fr-FR"),
        ("German", "de-DE"),
        ("Chinese (Mandarin)", "zh-CN")
    ], format_func=lambda x: x[0])[1]

    if st.button("Start Recording"):
        try:
            transcription = transcribe_speech(api_choice, language)
            st.success("Transcription complete!")
            st.text_area("Transcribed Text", transcription, height=200)

            if st.button("Save to File"):
                with open("transcription.txt", "w", encoding="utf-8") as f:
                    f.write(transcription)
                st.success("Saved to 'transcription.txt'")
        except Exception as e:
            st.error(f"Something went wrong: {e}")

if __name__ == "__main__":
    main()
