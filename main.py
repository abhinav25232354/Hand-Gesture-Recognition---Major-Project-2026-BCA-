import cv2
import mediapipe as mp

# 1. Initialize the standard legacy MediaPipe Hands API
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# 2. Start capturing video from webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Hand Gesture Recognition started. Press 'q' to quit.")

# 3. Use a context manager to initialize the Hands model
with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the frame horizontally for a mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB (MediaPipe requires RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # To improve performance, mark the image as not writeable to pass by reference
        image.flags.writeable = False
        
        # Process the image and find hands
        results = hands.process(image)
        
        # Mark the image as writeable again
        image.flags.writeable = True
        
        # Convert the image back to BGR for OpenCV rendering
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Optional: Extract landmark positions
                # for id, lm in enumerate(hand_landmarks.landmark):
                #     h, w, c = image.shape
                #     cx, cy = int(lm.x * w), int(lm.y * h)
                #     print(f"Landmark {id}: ({cx}, {cy})")
        
        # Display the output frame
        cv2.imshow('Hand Gesture Recognition', image)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 4. Release resources
cap.release()
cv2.destroyAllWindows()