import asyncio
import json
import aiohttp
import os
import tempfile
import speech_recognition as sr
from faster_whisper import WhisperModel
import winsound

async def listen_for_audio(queue: asyncio.Queue):
    """
    (The Ears)
    Uses SpeechRecognition to capture mic audio when speaking, 
    and Faster-Whisper (base.en) on the CPU to quickly transcribe it.
    """
    print("[STT Engine] Loading Faster-Whisper base.en model on CPU...")
    try:
        # Load model (downloads automatically on first run)
        model = WhisperModel("base.en", device="cpu", compute_type="int8")
        recognizer = sr.Recognizer()
        
        # Synchronous audio recording function to safely run in a background thread
        def record_audio():
            with sr.Microphone() as source:
                print("\n[Mic] Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print("[Mic] Listening... (Speak now!)")
                return recognizer.listen(source, phrase_time_limit=10)

        # Run the recording in a thread pool so it doesn't block the async loop
        audio_data = await asyncio.to_thread(record_audio)
        print("[Mic] Audio captured! Transcribing with Faster-Whisper...")
        
        # Save to a temporary WAV file for Whisper to read
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav.write(audio_data.get_wav_data())
            temp_path = temp_wav.name
            
        # Transcribe
        segments, info = model.transcribe(temp_path, beam_size=5)
        transcribed_text = "".join([segment.text for segment in segments]).strip()
        
        # Clean up temp file
        os.remove(temp_path)
        
        if transcribed_text:
            print(f"[STT] Final Text: '{transcribed_text}'")
            await queue.put(transcribed_text)
        else:
            print("[STT] No speech detected.")
            
    except sr.WaitTimeoutError:
        print("[STT] Timeout. No speech detected.")
    except Exception as e:
        print(f"\n[STT ERROR] Failed to record or transcribe. Error type: {type(e).__name__} - {e}")
        await asyncio.sleep(1)

async def process_llm(input_queue: asyncio.Queue, action_queue: asyncio.Queue, tts_queue: asyncio.Queue):
    """
    (The Brain)
    Takes transcribed text, sends it to Ollama (Llama 3.2), and parses the result.
    Routes to actions if JSON, routes to TTS if conversational text.
    """
    while True:
        text = await input_queue.get()
        print(f"[LLM] Pondering intent for: '{text}'...")
        
        prompt = f"""You are Friday, a highly capable AI assistant running on local hardware.
You have the ability to run system commands on a Windows machine. If the user asks you to perform a system action (like opening an app, managing files, checking stats, etc.), output exactly this JSON and nothing else: {{"command": "run_system_command", "target": "<the_windows_cmd_command>"}}
For example, to open the terminal: {{"command": "run_system_command", "target": "start cmd"}}
To open notepad: {{"command": "run_system_command", "target": "start notepad"}}
For any other conversational request, simply reply naturally and concisely.
User: {text}"""

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2",
                        "prompt": prompt,
                        "stream": False
                    }
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        llm_response = data.get("response", "").strip()
                        print(f"[LLM] Generated response: {llm_response}")
                        
                        try:
                             # Try parsing as JSON to see if it's an action
                             action_payload = json.loads(llm_response)
                             # Check if it's a valid action dict
                             if isinstance(action_payload, dict) and "command" in action_payload:
                                 await action_queue.put(action_payload)
                             else:
                                 await tts_queue.put(llm_response)
                        except json.JSONDecodeError:
                             # Not JSON, conversational text.
                             await tts_queue.put(llm_response)
                    else:
                        print(f"[LLM] Error: Ollama API returned status {response.status}")
        except aiohttp.ClientError as e:
            print(f"[LLM] Failed to connect to Ollama. Is it running? Error: {e}")
        
        input_queue.task_done()

async def execute_action(action_queue: asyncio.Queue):
    """
    (The Hands)
    Receives validated JSON objects and executes actual Python commands or OS processes.
    """
    while True:
        action = await action_queue.get()
        print(f"\n[ACTION ENGINE] Executing: {action['command']} on target {action.get('target', 'unknown')}")
        
        if action['command'] == "run_system_command":
             target_cmd = action.get('target', '')
             if target_cmd:
                 print(f"[ACTION ENGINE] Running OS command: {target_cmd}")
                 os.system(target_cmd)
             else:
                 print("[ACTION ENGINE] Error: No target command provided.")
        elif action['command'] == "open_terminal":
             # Kept for backwards compatibility if the AI uses it
             print("[ACTION ENGINE] Launching system terminal...")
             os.system("start cmd")
             
        action_queue.task_done()

async def speak_tts(tts_queue: asyncio.Queue):
    """
    (The Mouth)
    Receives standard text sentences from the LLM and feeds them to Piper TTS.
    """
    
    # Path to the downloaded Piper models
    model_path = "models/en_US-lessac-medium.onnx"
    import glob
    for p in glob.glob("*.onnx") + glob.glob("models/*.onnx"):
        model_path = p
        break
    
    print("\n[TTS Engine] Checking for Piper ONNX models...")
    if not os.path.exists("models"):
        os.makedirs("models")
        print("[TTS Engine] Created models directory.")
        
    if not os.path.exists(model_path):
        print(f"[TTS WARNING] Model not found at '{model_path}'")
        print(" -> To hear voices, download the Piper ONNX file (e.g., en_US-lessac-medium) and place it in the models/ folder.")
    else:
        print("[TTS Engine] Piper initialized successfully.")

    while True:
        text = await tts_queue.get()
        print(f"\n[TTS] Synthesizing audio for: '{text}'")
        
        if os.path.exists(model_path):
            try:
                # Piper generates audio entirely locally
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                    out_path = temp_wav.name
                
                # Run piper as a background shell process to avoid Windows pipe issues
                piper_exe = ".\\piper.exe" if os.path.exists("piper.exe") else "piper"
                command = f'{piper_exe} -m {model_path} -f {out_path}'
                
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                await process.communicate(input=text.encode('utf-8'))
                
                # Play the standard WAV output natively in Windows
                print("[TTS] Playing voice...")
                await asyncio.to_thread(winsound.PlaySound, out_path, winsound.SND_FILENAME)
                
            except Exception as e:
                print(f"[TTS ERROR] Failed to synthesize or play audio: {e}")
            finally:
                if 'out_path' in locals() and os.path.exists(out_path):
                    try:
                        os.remove(out_path)
                    except:
                        pass
        else:
            print("[TTS Dummy Output] -> Simulated Speaking: " + text)
            
        tts_queue.task_done()

async def main():
    print("=== Booting Friday Core System ===\n")
    
    # The central Hub pipelines
    input_queue = asyncio.Queue()
    action_queue = asyncio.Queue()
    tts_queue = asyncio.Queue()

    # Spin up the background daemons
    tasks = [
        asyncio.create_task(process_llm(input_queue, action_queue, tts_queue)),
        asyncio.create_task(execute_action(action_queue)),
        asyncio.create_task(speak_tts(tts_queue))
    ]

    print("\n[System] Friday is now active. Press Ctrl+C in the terminal to exit.")
    
    try:
        while True:
            # Continuously listen and process
            await listen_for_audio(input_queue)
            
            # Wait for Friday to finish processing and speaking before listening again
            await input_queue.join()
            await action_queue.join()
            await tts_queue.join()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass

    # Shut down cleanly
    for task in tasks:
        task.cancel()
    
    print("\n=== System Shutdown. ===")

if __name__ == "__main__":
    asyncio.run(main())
