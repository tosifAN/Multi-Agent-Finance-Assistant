import os
import json
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
from datetime import datetime
import tempfile

# Import necessary modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define request and response models
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "en-US-Neural2-F"  # Default voice
    parameters: Optional[Dict[str, Any]] = None

class STTRequest(BaseModel):
    audio_file_path: str
    parameters: Optional[Dict[str, Any]] = None

class VoiceResponse(BaseModel):
    status: str
    data: Dict[str, Any]
    timestamp: str

# Create the FastAPI app
app = FastAPI(title="Voice Agent", description="Agent for handling speech-to-text and text-to-speech conversions")

class VoiceAgent:
    """Agent for handling speech-to-text and text-to-speech conversions"""
    
    def __init__(self):
        """Initialize the voice agent"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
    
    async def text_to_speech(self, text: str, voice: str = "en-US-Neural2-F", parameters: Optional[Dict] = None) -> Dict:
        """Convert text to speech
        
        Args:
            text: Text to convert to speech
            voice: Voice to use for speech
            parameters: Optional parameters for the conversion
            
        Returns:
            Dictionary containing the speech data
        """
        try:
            import pyttsx3
        except ImportError:
            raise ImportError("Please install pyttsx3: pip install pyttsx3")
        
        # Create a temporary file for the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file_path = temp_file.name
        temp_file.close()
        
        # Initialize the TTS engine
        engine = pyttsx3.init()
        
        # Set properties
        rate = parameters.get("rate", 150) if parameters else 150
        volume = parameters.get("volume", 1.0) if parameters else 1.0
        
        engine.setProperty("rate", rate)
        engine.setProperty("volume", volume)
        
        # Get available voices and set the voice
        voices = engine.getProperty("voices")
        for v in voices:
            if voice in v.id:
                engine.setProperty("voice", v.id)
                break
        
        # Convert text to speech and save to file
        engine.save_to_file(text, temp_file_path)
        engine.runAndWait()
        
        return {
            "audio_file_path": temp_file_path,
            "text": text,
            "voice": voice
        }
    
    async def speech_to_text(self, audio_file_path: str, parameters: Optional[Dict] = None) -> Dict:
        """Convert speech to text
        
        Args:
            audio_file_path: Path to the audio file
            parameters: Optional parameters for the conversion
            
        Returns:
            Dictionary containing the transcription data
        """
        try:
            import whisper
        except ImportError:
            raise ImportError("Please install whisper: pip install openai-whisper")
        
        # Load the model
        model_name = parameters.get("model", "base") if parameters else "base"
        model = whisper.load_model(model_name)
        
        # Transcribe the audio
        result = model.transcribe(audio_file_path)
        
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"],
            "audio_file_path": audio_file_path
        }
    
    async def process_tts_request(self, request: TTSRequest) -> Dict:
        """Process a text-to-speech request
        
        Args:
            request: TTSRequest object
            
        Returns:
            Dictionary containing the response data
        """
        try:
            # Convert text to speech
            result = await self.text_to_speech(
                text=request.text,
                voice=request.voice,
                parameters=request.parameters
            )
            
            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "data": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
    
    async def process_stt_request(self, request: STTRequest) -> Dict:
        """Process a speech-to-text request
        
        Args:
            request: STTRequest object
            
        Returns:
            Dictionary containing the response data
        """
        try:
            # Convert speech to text
            result = await self.speech_to_text(
                audio_file_path=request.audio_file_path,
                parameters=request.parameters
            )
            
            return {
                "status": "success",
                "data": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "data": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }

# Dependency to get the voice agent
def get_voice_agent():
    return VoiceAgent()

# API routes
@app.post("/api/tts", response_model=VoiceResponse)
async def text_to_speech(request: TTSRequest, 
                       voice_agent: VoiceAgent = Depends(get_voice_agent)):
    """Endpoint for converting text to speech"""
    response = await voice_agent.process_tts_request(request)
    return response

@app.post("/api/stt", response_model=VoiceResponse)
async def speech_to_text(request: STTRequest, 
                       voice_agent: VoiceAgent = Depends(get_voice_agent)):
    """Endpoint for converting speech to text"""
    response = await voice_agent.process_stt_request(request)
    return response

@app.post("/api/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    """Endpoint for uploading audio files"""
    # Create a temporary file to store the uploaded audio
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file_path = temp_file.name
    
    # Write the uploaded file to the temporary file
    content = await file.read()
    with open(temp_file_path, "wb") as f:
        f.write(content)
    
    return {
        "status": "success",
        "data": {"audio_file_path": temp_file_path},
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Run the FastAPI app if this module is executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)