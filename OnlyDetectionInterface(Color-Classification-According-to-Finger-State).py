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
        finger_count = 0
        mode_text = "Normal"

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Determine handedness label for thumb logic
                hand_label = None
                if results.multi_handedness and len(results.multi_handedness) > idx:
                    hand_label = results.multi_handedness[idx].classification[0].label

                # Count fingers for this hand
                lm = hand_landmarks.landmark
                tips_ids = [4, 8, 12, 16, 20]
                fingers = []

                # Thumb
                if hand_label:
                    if hand_label == 'Right':
                        fingers.append(1 if lm[4].x < lm[3].x else 0)
                    else:
                        fingers.append(1 if lm[4].x > lm[3].x else 0)
                else:
                    # Fallback: use relative x to detect thumb
                    fingers.append(1 if lm[4].x < lm[3].x else 0)

                # Other four fingers: compare tip to pip
                for tip_id in tips_ids[1:]:
                    if lm[tip_id].y < lm[tip_id - 2].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)

                # Use finger count of first detected hand (ignore additional hands)
                finger_count = sum(fingers)
                break
        
        # Apply color/hue changes according to finger_count
        if finger_count == 1:
            # Greyscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            mode_text = "Greyscale"
        elif finger_count == 2:
            # Purple tint (set hue to purple)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            h[:] = 140
            hsv = cv2.merge([h, s, v])
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            mode_text = "Purple"
        elif finger_count == 3:
            # Orange tint
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            h[:] = 10
            hsv = cv2.merge([h, s, v])
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            mode_text = "Orange"
        elif finger_count == 4:
            # Yellow tint
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            h[:] = 25
            hsv = cv2.merge([h, s, v])
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            mode_text = "Yellow"
        else:
            # Normal (no change) when 0 or 5+
            mode_text = "Normal"

        # Overlay current finger count and mode
        cv2.rectangle(image, (0,0), (220,40), (0,0,0), -1)
        cv2.putText(image, f'Fingers: {finger_count}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(image, f'Mode: {mode_text}', (10,36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Display the output frame
        cv2.imshow('Hand Gesture Recognition', image)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 4. Release resources
cap.release()
cv2.destroyAllWindows()