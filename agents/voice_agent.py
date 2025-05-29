import os
import tempfile
import numpy as np
from typing import Dict, Any, Optional, Tuple
from crewai import Agent, Task
import whisper
from gtts import gTTS
import pyttsx3
from pydub import AudioSegment
from pydub.playback import play

class VoiceAgent:
    """Agent for handling speech-to-text and text-to-speech operations."""
    
    def __init__(self, whisper_model: str = None, tts_engine: str = None):
        """Initialize the voice agent.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            tts_engine: Text-to-speech engine (gtts, pyttsx3)
        """
        self.whisper_model = whisper_model or os.getenv('WHISPER_MODEL', 'base')
        self.tts_engine = tts_engine or os.getenv('TTS_ENGINE', 'gtts')
        
        # Load Whisper model
        print(f"Loading Whisper model: {self.whisper_model}")
        self.model = whisper.load_model(self.whisper_model)
        
        # Initialize TTS engine if using pyttsx3
        if self.tts_engine == 'pyttsx3':
            self.tts = pyttsx3.init()
        
    def create_agent(self) -> Agent:
        """Create a CrewAI agent for voice operations."""
        return Agent(
            role="Voice Processing Specialist",
            goal="Convert between speech and text accurately and naturally",
            backstory="""You are an expert in voice processing technologies with years of 
            experience in speech recognition and synthesis. Your specialty is in 
            converting spoken language to text and generating natural-sounding speech 
            from text input.""",
            verbose=True,
            allow_delegation=False
        )
    
    def transcribe_audio(self, audio_file: str) -> Dict[str, Any]:
        """Transcribe audio file to text using Whisper.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Dictionary with transcription results
        """
        try:
            print("you are under voice agent transcribe_audio fucntion ")

            # Transcribe audio
            result = self.model.transcribe(audio_file)
            print(f"this is audio file path ${audio_file}")

            audio_file = os.path.abspath(audio_file)

            if not os.path.exists(audio_file):
                logger.error(f"File not found: {audio_file}")
                raise FileNotFoundError(f"The audio file does not exist at path: {audio_file}")
            print(f"this is audio file path ${audio_file}")
            print(f"this is the result ahahha : ${result}")
            
            return {
                'text': result['text'],
                'segments': result['segments'],
                'language': result['language'],
                'success': True
            }
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return {
                'text': '',
                'success': False,
                'error': str(e)
            }
    
    def text_to_speech(self, text: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Convert text to speech.
        
        Args:
            text: Text to convert to speech
            output_file: Path to save audio file (optional)
            
        Returns:
            Dictionary with TTS results
        """
        try:
            # Generate temporary file if output_file not provided
            if output_file is None:
                temp_dir = tempfile.gettempdir()
                output_file = os.path.join(temp_dir, 'tts_output.mp3')
            
            # Use selected TTS engine
            if self.tts_engine == 'gtts':
                tts = gTTS(text=text, lang='en', slow=False)
                tts.save(output_file)
            elif self.tts_engine == 'pyttsx3':
                # pyttsx3 saves as .wav
                wav_file = output_file.replace('.mp3', '.wav')
                self.tts.save_to_file(text, wav_file)
                self.tts.runAndWait()
                
                # Convert to mp3 if needed
                if output_file.endswith('.mp3'):
                    sound = AudioSegment.from_wav(wav_file)
                    sound.export(output_file, format="mp3")
                    os.remove(wav_file)  # Clean up temporary wav file
            else:
                raise ValueError(f"Unsupported TTS engine: {self.tts_engine}")
            
            return {
                'output_file': output_file,
                'text': text,
                'success': True
            }
        except Exception as e:
            print(f"Error converting text to speech: {e}")
            return {
                'output_file': None,
                'text': text,
                'success': False,
                'error': str(e)
            }
    
    def play_audio(self, audio_file: str) -> bool:
        """Play audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Boolean indicating success
        """
        try:
            sound = AudioSegment.from_file(audio_file)
            play(sound)
            return True
        except Exception as e:
            print(f"Error playing audio: {e}")
            return False
    
    def process_voice_query(self, audio_file: str) -> Tuple[str, Dict[str, Any]]:
        """Process a voice query by transcribing it to text.
        
        Args:
            audio_file: Path to audio file containing the query
            
        Returns:
            Tuple of (query_text, transcription_details)
        """
        transcription = self.transcribe_audio(audio_file)
        query_text = transcription['text']
        
        return query_text, transcription
    
    def deliver_voice_response(self, response_text: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Deliver a response as speech.
        
        Args:
            response_text: Text to convert to speech
            output_file: Path to save audio file (optional)
            
        Returns:
            Dictionary with TTS results
        """
        tts_result = self.text_to_speech(response_text, output_file)
        
        if tts_result['success'] and tts_result['output_file']:
            # Automatically play the response
            self.play_audio(tts_result['output_file'])
        
        return tts_result

# Example tasks for the voice agent
def create_voice_tasks(agent: Agent) -> list[Task]:
    """Create tasks for the voice agent."""
    return [
        Task(
            description="Transcribe a voice query about Asia tech stocks",
            agent=agent,
            expected_output="Accurate text transcription of the voice query"
        ),
        Task(
            description="Convert a market brief to natural-sounding speech",
            agent=agent,
            expected_output="An audio file containing the spoken market brief with natural intonation and clarity"
        ),
        Task(
            description="Process a voice query and deliver a spoken response",
            agent=agent,
            expected_output="End-to-end processing from voice input to voice output with high accuracy and natural sound"
        )
    ]