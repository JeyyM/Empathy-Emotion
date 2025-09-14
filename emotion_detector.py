# Installs:
# pip install opencv-python
# pip install deepface
# pip install numpy

import cv2
from deepface import DeepFace
import numpy as np
from collections import deque
import threading

# For voice emotion recognition
# pip install transformers torchaudio sounddevice
import sounddevice as sd
import torch
import torchaudio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor

# Load Hugging Face voice emotion model and feature extractor
VOICE_MODEL_NAME = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
voice_model = None
voice_feature_extractor = None
voice_emotion_labels = [
    'angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'
]

def load_voice_model():
    global voice_model, voice_feature_extractor
    if voice_model is None or voice_feature_extractor is None:
        voice_model = Wav2Vec2ForSequenceClassification.from_pretrained(VOICE_MODEL_NAME)
        voice_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(VOICE_MODEL_NAME)

def record_audio(duration=3, fs=16000):
    print(f"Recording {duration} seconds of audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(audio)
    return audio, fs

def predict_voice_emotion(audio, fs):
    load_voice_model()
    if fs != 16000:
        audio = torchaudio.functional.resample(torch.tensor(audio), fs, 16000).numpy()
        fs = 16000
    inputs = voice_feature_extractor(audio, sampling_rate=fs, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = voice_model(**inputs).logits
    predicted_id = int(torch.argmax(logits, dim=1))
    emotion = voice_emotion_labels[predicted_id]
    confidence = torch.softmax(logits, dim=1)[0, predicted_id].item() * 100
    return emotion.title(), confidence

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Some OpenCV landmarkers
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    print("Camera opened successfully!")
    print("Press 'q' to quit")
    print("Press 'l' to toggle landmarks")
    print("Voice emotion will be detected automatically.")
    
    frame_count = 0
    current_emotion = "Detecting"
    current_emotion_confidence = 0.0
    show_landmarks = False
    last_face_region = None
    
    # Emotion selection from most recent, without this, emotion is jittery
    emotion_history = deque(maxlen=5)  # Keep last 5 emotion readings
    emotion_confidence_threshold = 40.0  # Only accept emotions above this confidence

    # For voice emotion
    last_voice_emotion = None
    last_voice_confidence = None

    # Start background thread for continuous voice emotion recognition
    def continuous_voice_emotion():
        nonlocal last_voice_emotion, last_voice_confidence
        while True:
            audio, fs = record_audio(duration=3, fs=16000)
            try:
                voice_emotion, voice_conf = predict_voice_emotion(audio, fs)
                last_voice_emotion = voice_emotion
                last_voice_confidence = voice_conf
                print(f"Voice Emotion: {voice_emotion} ({voice_conf:.1f}%)")
            except Exception as e:
                print(f"Voice emotion detection failed: {e}")
                last_voice_emotion = "Error"
                last_voice_confidence = None

    threading.Thread(target=continuous_voice_emotion, daemon=True).start()

    while True:
        # To capture camera video
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        if frame_count % 15 == 0:
            try:
                # DeepFace pre-trained data
                result = DeepFace.analyze(
                    frame, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    detector_backend='opencv',
                    silent=True
                )

                # To only track one face
                if isinstance(result, list):
                    result = result[0] 

                emotions = result['emotion']
                dominant_emotion = result['dominant_emotion']
                dominant_confidence = emotions[dominant_emotion]

                # Only update emotion if confidence is high enough
                if dominant_confidence > emotion_confidence_threshold:
                    emotion_history.append(dominant_emotion)

                    # Get most common emotion from past frames
                    if len(emotion_history) >= 3:
                        emotion_counts = {}
                        for emotion in emotion_history:
                            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                        stable_emotion = max(emotion_counts, key=emotion_counts.get)
                        current_emotion = stable_emotion.title()
                        current_emotion_confidence = emotions[stable_emotion]
                    else:
                        current_emotion = dominant_emotion.title()
                        current_emotion_confidence = dominant_confidence

                # To show landmark
                face_region = result['region']
                x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']

                # Store face region for landmark drawing
                last_face_region = (x, y, w, h)

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display dominant emotion with confidence on landmark
                cv2.putText(frame, f"Emotion: {current_emotion} ({current_emotion_confidence:.1f}%)", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Display top 3 emotion confidence scores only
                sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                y_offset = y + h + 20
                for emotion, confidence in sorted_emotions:
                    color = (0, 255, 0) if emotion == dominant_emotion else (255, 255, 255)
                    text = f"{emotion}: {confidence:.1f}%"
                    cv2.putText(frame, text, (x, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_offset += 20

            except Exception as e:
                # If no face is detected or other error, just show the frame
                cv2.putText(frame, "No face detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                current_emotion = "No Face"
                current_emotion_confidence = 0.0
                last_face_region = None
        
        # Toggle landmarks
        if show_landmarks and last_face_region is not None:
            x, y, w, h = last_face_region
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # Extract face region for eye detection
            face_roi = frame[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Feature points
            # Detect eyes
            eyes = eye_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
            
            for (ex, ey, ew, eh) in eyes:
                # Convert eye coordinates to full frame coordinates
                eye_x = x + ex + ew//2
                eye_y = y + ey + eh//2
                cv2.circle(frame, (eye_x, eye_y), 15, (255, 0, 0), 2)
                cv2.circle(frame, (eye_x, eye_y), 3, (255, 0, 0), -1)
            
            # Detect nose
            nose_x = x + w//2
            nose_y = y + int(h * 0.65)
            cv2.circle(frame, (nose_x, nose_y), 3, (0, 255, 0), -1)
            
            # Detect mouth corners
            mouth_left = (x + int(w * 0.3), y + int(h * 0.8))
            mouth_right = (x + int(w * 0.7), y + int(h * 0.8))
            cv2.circle(frame, mouth_left, 3, (0, 0, 255), -1)
            cv2.circle(frame, mouth_right, 3, (0, 0, 255), -1)
        
        # Write emotion tracker in corner
        tracker_text = f"Facial: {current_emotion}"
        if current_emotion_confidence > 0:
            tracker_text += f" ({current_emotion_confidence:.1f}%)"
        landmarks_status = "ON" if show_landmarks else "OFF"
        tracker_text += f" | Landmarks: {landmarks_status}"
        if last_voice_emotion is not None:
            tracker_text += f" | Voice: {last_voice_emotion}"
            if last_voice_confidence is not None:
                tracker_text += f" ({last_voice_confidence:.1f}%)"

        # Set smaller font size and thickness
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(tracker_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Position for top-right corner
        x_pos = frame_width - text_size[0] - 10
        y_pos = 20

        # Draw background rectangle with less padding
        cv2.rectangle(frame, (x_pos - 5, y_pos - 18),
                     (x_pos + text_size[0] + 5, y_pos + 8),
                     (0, 0, 0), -1)

        # Draw the tracker text
        cv2.putText(frame, tracker_text, (x_pos, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), font_thickness)
        
        # Display the resulting frame
        cv2.imshow('Emotion Detection', frame)
        
        frame_count += 1
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            show_landmarks = not show_landmarks
            print(f"Landmarks {'ON' if show_landmarks else 'OFF'}")
        # No need for 'v' key, voice emotion runs in background
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed")

if __name__ == "__main__":
    main()
