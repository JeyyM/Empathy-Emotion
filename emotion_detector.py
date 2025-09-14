import cv2
from deepface import DeepFace
import numpy as np
from collections import deque

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
    
    frame_count = 0
    current_emotion = "Detecting..."
    show_landmarks = False
    last_face_region = None
    
    # Emotion smoothing, without this, emotion is jittery
    emotion_history = deque(maxlen=5)  # Keep last 5 emotion readings
    emotion_confidence_threshold = 40.0  # Only accept emotions above this confidence
    
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
                    else:
                        current_emotion = dominant_emotion.title()
                
                # To show landmark
                face_region = result['region']
                x, y, w, h = face_region['x'], face_region['y'], face_region['w'], face_region['h']
                
                # Store face region for landmark drawing
                last_face_region = (x, y, w, h)
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Display dominant emotion with confidence on landmark
                cv2.putText(frame, f"Emotion: {current_emotion} ({dominant_confidence:.1f}%)", 
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
        tracker_text = f"Current: {current_emotion}"
        landmarks_status = "ON" if show_landmarks else "OFF"
        tracker_text += f" | Landmarks: {landmarks_status}"
        text_size = cv2.getTextSize(tracker_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Position for top-right corner
        x_pos = frame_width - text_size[0] - 20
        y_pos = 30
        
        # Text background rectangle
        cv2.rectangle(frame, (x_pos - 10, y_pos - 25), 
                     (x_pos + text_size[0] + 10, y_pos + 10), 
                     (0, 0, 0), -1)
        
        # Write emotion tracker text
        cv2.putText(frame, tracker_text, (x_pos, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
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
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera released and windows closed")

if __name__ == "__main__":
    main()
