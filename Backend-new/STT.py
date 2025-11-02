import sounddevice as sd
from scipy.io.wavfile import write
import speech_recognition as sr
from googletrans import Translator
import time
import os

# --- Configuration ---
FS = 44100  # Sample rate (standard CD quality)
SECONDS = 10 # Duration of recording in seconds
FILENAME = "hindi_audio.wav"

def record_audio(filename, fs, seconds):
    """
    Records audio from the microphone and saves it as a WAV file.
    """
    print(f"🎤 Recording for {seconds} seconds. Please speak in HINDI now...")
    # sd.rec records the audio
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=2, dtype='int16')
    # sd.wait waits until recording is finished
    sd.wait()
    # write saves the numpy array (recording) to a WAV file
    write(filename, fs, recording)
    print(f"✅ Recording complete! Saved to {filename}")

def transcribe_and_translate_audio(filename):
    """
    Transcribes Hindi audio to Hindi text, then translates the text to English.
    """
    r = sr.Recognizer()
    translator = Translator() # Initialize the Google Translator
    
    if not os.path.exists(filename):
        print(f"🚨 Error: File not found at {filename}")
        return

    with sr.AudioFile(filename) as source:
        print("\n⏳ Analyzing audio file...")
        # Load the entire audio file into an AudioData object
        audio = r.record(source)  
    
    # --- Step 1: Transcribe Hindi Speech to Hindi Text ---
    try:
        print("🚀 Transcribing Hindi audio to Hindi text...")
        # Specify the language code for the speech you expect: Hindi (India)
        text_hindi = r.recognize_google(audio, language='hi-IN')
        print(f"\n[HINDI TRANSCRIPTION]: {text_hindi}")
        
    except sr.UnknownValueError:
        print("\n❌ Could not understand audio (Hindi). Please speak clearly.")
        return
    except sr.RequestError as e:
        print(f"\n❌ Transcription request failed (Check internet/quota); {e}")
        return

    # --- Step 2: Translate Hindi Text to English Text ---
    try:
        print("🌐 Translating Hindi text to English...")
        
        # Translate the Hindi text (source='auto' detects Hindi) to English (dest='en')
        translation = translator.translate(text_hindi, dest='en')
        text_english = translation.text
        
        print("\n" + "="*60)
        print("            ✨ English Translation Output ✨")
        print("="*60)
        print(f"ENGLISH TRANSLATION: {text_english}")
        print("="*60 + "\n")

    except Exception as e:
        print(f"❌ Translation failed; {e}")

# --- Main execution block ---
if __name__ == "__main__":
    
    print("\n--- Python Hindi Speech-to-English Translator ---")
    
    # 1. Record the WAV file
    record_audio(FILENAME, FS, SECONDS)
    
    # Small pause for file system to catch up
    time.sleep(1) 
    
    # 2. Decode and Translate the WAV file
    transcribe_and_translate_audio(FILENAME)