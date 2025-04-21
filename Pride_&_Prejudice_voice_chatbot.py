# Import necessary libraries
import nltk  # Natural Language Toolkit for text processing
import streamlit as st  # Streamlit for building the web app interface
from nltk.tokenize import word_tokenize, sent_tokenize  # Tokenizing sentences and words
from nltk.corpus import stopwords  # Stopwords list to remove common words like "the", "and"
from nltk.stem import WordNetLemmatizer  # Lemmatizer for converting words to their root form
import string  # String for handling punctuation
import speech_recognition as sr  # Speech recognition library for converting speech to text

# Initialize the speech recognition engine
recognizer = sr.Recognizer()

# Load stopwords (common words to be removed) and initialize the lemmatizer (for root word extraction)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Step 1: Function to transcribe live speech input using the microphone
def transcribe_speech():
    try:
        with sr.Microphone() as source:  # Use the microphone as the audio source
            st.info("Speak now... Please speak clearly!")
            audio = recognizer.listen(source)  # Capture the speech from the microphone
            st.info("Transcribing speech...")
            return recognizer.recognize_google(audio)  # Transcribe speech to text using Google API
    except sr.UnknownValueError:  # Error handling if speech is not recognized
        return "Sorry, I couldn't understand that."
    except sr.RequestError:  # Error handling for API unavailability
        return "API unavailable or unresponsive."
    except Exception as e:  # Generic error handling
        return f"An error occurred: {str(e)}"

# Step 2: Function to transcribe an audio file input
def transcribe_audio_file(file_path):
    try:
        with sr.AudioFile(file_path) as source:  # Use the uploaded audio file as the source
            audio = recognizer.record(source)  # Record the entire file
            st.info("Transcribing audio file...")
            return recognizer.recognize_google(audio)  # Transcribe the audio to text using Google API
    except sr.UnknownValueError:  # Error handling for unrecognized audio
        return "Sorry, I couldn't understand the audio."
    except sr.RequestError:  # Error handling if the API is unresponsive
        return "API unavailable or unresponsive."
    except Exception as e:  # Generic error handling
        return f"An error occurred: {str(e)}"
    
# Text preprocessing function: tokenizes, removes stopwords, lemmatizes, and removes punctuation
def preprocess(sentence):
    words = word_tokenize(sentence.lower())  # Tokenize the sentence into words and convert to lowercase
    words = [word for word in words if word not in stop_words and word not in string.punctuation]  # Remove stopwords and punctuation
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize the words to their root forms
    return words

# Function to load and preprocess the text file (Pride and Prejudice)
def load_text():
    try:
        file_path = r"C:\\Users\\braid\\Downloads\\Pride and Prejudice.txt"  # Path to the text file
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().replace('\\n', ' ')  # Load and return the text, replacing line breaks with spaces
    except FileNotFoundError:  # Error handling if the file is not found
        st.error("Text file not found.")
        return ""

# Tokenizes the text into sentences and preprocesses each sentence
def prepare_corpus(text):
    sentences = sent_tokenize(text)  # Split the text into individual sentences
    return [preprocess(sentence) for sentence in sentences]  # Preprocess each sentence

# Function to calculate Jaccard similarity between two sets of words
def jaccard_similarity(query, sentence):
    query_set = set(query)  # Convert the query to a set of unique words
    sentence_set = set(sentence)  # Convert the sentence to a set of unique words
    if len(query_set.union(sentence_set)) == 0:  # If both sets are empty
        return 0
    return len(query_set.intersection(sentence_set)) / len(query_set.union(sentence_set))  # Calculate Jaccard similarity

# Function to find the most relevant sentence based on Jaccard similarity
def get_most_relevant_sentence(query, corpus, original_sentences):
    query = preprocess(query)  # Preprocess the user query
    max_similarity = 0  # Initialize maximum similarity
    best_sentence = "I couldn't find a relevant answer."  # Default response if no relevant sentence is found
    for i, sentence in enumerate(corpus):  # Iterate through each sentence in the corpus
        similarity = jaccard_similarity(query, sentence)  # Calculate similarity between query and the current sentence
        if similarity > max_similarity:  # If the current sentence has higher similarity than the previous max
            max_similarity = similarity
            best_sentence = original_sentences[i]  # Set the best sentence to the original sentence with the highest similarity
    return best_sentence

# Main function to create the Streamlit interface for the chatbot
def main():
    st.title("Pride and Prejudice's Novice Chatbot with Voice Input")  # App title
    st.write("Ask me anything related to Pride and Prejudice! You can either speak, upload an audio file, or type your question.")  # App description

    # Suggestions for users to try asking
    with st.expander("Click me for suggestions"):
        st.write("""
        1. What is the first line of Pride and Prejudice?
        2. What is Mr. Bennet’s response when Mrs. Bennet urges him to visit Mr. Bingley?
        3. How many daughters do Mr. and Mrs. Bennet have?
        4. What is Mr. Bingley's annual income, according to the neighborhood gossip?
        5. What does Mr. Darcy say about Elizabeth at the Meryton assembly?
        """)

    # Load the text of Pride and Prejudice and prepare the corpus
    text = load_text()  # Load the text
    if text:
        corpus = prepare_corpus(text)  # Preprocess the corpus (tokenized and cleaned)
        original_sentences = sent_tokenize(text)  # Store the original sentences

        # User input options: live speech, file upload, or text input
        speech_input = st.button("Speak your question")  # Button for live speech input
        file_input = st.file_uploader("Upload an audio file:", type=['wav', 'mp3'])  # Upload an audio file
        user_input = st.text_input("Or type your question:")  # Text input option

        # Handle live speech input
        if speech_input:
            user_input = transcribe_speech()  # Transcribe the speech to text
            st.write(f"Transcribed Text: {user_input}")  # Display the transcribed text

        # Handle audio file input
        if file_input:
            user_input = transcribe_audio_file(file_input)  # Transcribe the audio file to text
            st.write(f"Transcribed Audio File: {user_input}")  # Display the transcribed text

        # Process the user input if available
        if user_input:
            response = get_most_relevant_sentence(user_input, corpus, original_sentences)  # Get the most relevant sentence
            st.write(f"Chatbot: {response}")  # Display the chatbot's response
        else:
            st.write("Please ask a question by speaking, uploading a file, or typing.")  # Prompt the user to ask a question

# Run the Streamlit app
if __name__ == "__main__":  # If the script is being run directly
    main()  # Run the main function
    
