#Finalised code
import os
import io
import pygame
import streamlit as st
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
from gtts import gTTS
from langdetect import detect
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import requests
import pyaudio
import wave

# Load the environment variables
load_dotenv()

# Embeddings model name
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# Define the model name
llm_model = "Llama3-8b-8192"

embeddings = HuggingFaceEmbeddings(model_name=model_name)

# groq api key
groq_api_key = os.environ.get('GROQ_API_KEY')

def translate(answer):
    try:
        detected_lang = detect(answer)
        if detected_lang == 'en':
            translation_text = GoogleTranslator(source='en', target='kn').translate(answer)
        elif detected_lang == 'kn':
            translation_text = GoogleTranslator(source='kn', target='en').translate(answer)
        else:
            raise ValueError("Language not supported")
        return translation_text
    except Exception as e:
        st.error(f"Translation failed: {e}")

def tts(answer):
    try:
        detected_lang = detect(answer)
        tts_lang = 'en' if detected_lang == 'en' else 'kn'
        tts = gTTS(text=answer, lang=tts_lang)
        speech = io.BytesIO()
        tts.write_to_fp(speech)
        speech.seek(0)
        pygame.mixer.init()
        pygame.mixer.music.load(speech)
        pygame.mixer.music.play()
    except Exception as e:
        st.error(f"Text-to-speech conversion failed: {e}")

def user_prompt(user_input, retrieval_chain):
    if user_input:
        detected_lang = detect(user_input)
        if detected_lang == 'en':
            response_content = retrieval_chain.invoke({"input": user_input})
            answer = response_content["answer"]
            st.session_state[f"answer_{st.session_state.chat_counter}"] = answer
            st.session_state[f"original_answer_{st.session_state.chat_counter}"] = answer  # Store original response
            #st.text("Response:")
            #st.write(answer)
        elif detected_lang == 'kn':
            user_input_kan = translate(user_input)
            response_content = retrieval_chain.invoke({"input": user_input_kan})
            answer = response_content["answer"]
            answer = translate(answer)
            st.session_state[f"answer_{st.session_state.chat_counter}"] = answer
            st.session_state[f"original_answer_{st.session_state.chat_counter}"] = answer  # Store original response
            #st.text("Response:")
            #st.write(answer)

def stt_eng():
    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-tiny"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

    def query_audio(audio_data):
        response = requests.post(API_URL, headers=headers, data=audio_data)
        return response.json().get("text", "")

    def record_audio(filename, duration=5, channels=1, rate=44100, chunk=1024):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=channels,
                            rate=rate, input=True,
                            frames_per_buffer=chunk)
        frames = []
        st.text("Recording...")
        for _ in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
        st.text("Finished recording.")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    filename = "user_audio.wav"
    record_audio(filename)
    with open(filename, "rb") as f:
        audio_data = f.read()
    generated_text = query_audio(audio_data)
    
    # Assign the generated text to the correct session state key
    st.session_state.generated_text = generated_text
    st.experimental_rerun()
    
def stt_kan():
    API_URL = "https://api-inference.huggingface.co/models/vasista22/whisper-kannada-medium"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}

    def query_audio(audio_data):
        response = requests.post(API_URL, headers=headers, data=audio_data)
        return response.json().get("text", "")

    def record_audio(filename, duration=5, channels=1, rate=44100, chunk=1024):
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=channels,
                            rate=rate, input=True,
                            frames_per_buffer=chunk)
        frames = []
        st.text("Recording...")
        for _ in range(0, int(rate / chunk * duration)):
            data = stream.read(chunk)
            frames.append(data)
        st.text("Finished recording.")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        wf = wave.open(filename, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()

    filename = "user_audio_kannada.wav"
    record_audio(filename)
    with open(filename, "rb") as f:
        audio_data = f.read()
    generated_text = query_audio(audio_data)
    
    # Assign the generated text to the correct session state key
    st.session_state.generated_text = generated_text
    st.experimental_rerun()


def main():
    if "chat_counter" not in st.session_state:
        st.session_state.chat_counter = 0
    if "clear_chat_confirm" not in st.session_state:
        st.session_state.clear_chat_confirm = False
    if "translated_answer" not in st.session_state:
        st.session_state.translated_answer = None  # Initialize translated answer

    try:
        # Initialize the GROQ model and prompt
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
            temperature=0.2
        )

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant specializing in fundamental rights within the judicial system. Your goal is to provide accurate information and guidance based on the documents provided to you. If you encounter a question outside the scope of the provided documents, simply respond with "I'm sorry, I don't have information on that topic." Use the following pieces of information to answer the user's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        <context>
        {context}
        </context>

        Question: {input}
        Only return the helpful answer below and nothing else.
        Helpful answer:
        """)

        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Load the FAISS index
        faiss_vector_store = FAISS.load_local("faiss_store", embeddings, allow_dangerous_deserialization=True)
        retriever = faiss_vector_store.as_retriever(search_kwargs={'k': 1})

        # Set up the retrieval chain
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Streamlit UI
        st.title("RightsMate 🕴️")
        st.write("Unlock the power of legal knowledge with RightsMate! 💼✨")

        st.sidebar.title("Legal Counsel Corner: Let's Chat Law 📚⚖️")

        if st.sidebar.button("🔄 New Chat"):
            st.session_state.chat_counter += 1
            st.session_state[f"user_input_{st.session_state.chat_counter}"] = ""
            st.session_state[f"answer_{st.session_state.chat_counter}"] = ""
            st.session_state.translated_answer = None  # Reset translated answer

        current_input_key = f"user_input_{st.session_state.chat_counter}"
        current_answer_key = f"answer_{st.session_state.chat_counter}"

        if st.sidebar.button("🗑️ Clear Chat"):
            st.session_state.clear_chat_confirm = True

        if st.session_state.clear_chat_confirm:
            if st.button("Are you sure you want to clear the chat?"):
                st.session_state[current_input_key] = ""
                st.session_state[current_answer_key] = ""
                st.session_state.clear_chat_confirm = False
                st.session_state.translated_answer = None  # Reset translated answer
                st.experimental_rerun()

        # Check if there is generated text from STT
        if "generated_text" in st.session_state:
            generated_text = st.session_state.pop("generated_text")
            st.session_state[current_input_key] = generated_text

        user_input = st.text_input("Inquire Within: Pose Your Legal Queries Below 👩‍⚖️💬:", key=current_input_key)
        
        if user_input:
            user_prompt(user_input, retrieval_chain)

        if st.session_state.get(current_answer_key):
            response = st.session_state.get(current_answer_key)
            original_response = st.session_state.get(f"original_answer_{st.session_state.chat_counter}")
            st.text("Response:")
            st.write(response)

            if st.session_state.translated_answer is None:
                st.sidebar.markdown("---")
                if st.sidebar.button("🔁 Translate"):
                    st.session_state.translated_answer = translate(response)
                    st.text("Translated Response:")
                    st.write(st.session_state.translated_answer)

                if st.sidebar.button("🔊 Text to Speech"):
                    tts(original_response)

        if st.sidebar.button("🎤 Speech to Text English"):
            stt_eng()

        if st.sidebar.button("🎙️ Speech to Text Kannada"):
            stt_kan()

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
