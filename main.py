import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Set page configuration
st.set_page_config(page_title="Virtual Air Canvas", layout="wide", initial_sidebar_state="collapsed")

# Enhanced CSS styles for better appearance
st.markdown("""
    <style>
        body {
            background-color: #e7f1ff; /* Light background color */
            font-family: 'Arial', sans-serif;
        }
        .main {
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        .title-text {
            font-size: 3em;
            color: #2f3542;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle-text {
            font-size: 1.5em;
            color: #576574;
            text-align: center;
            margin-bottom: 30px;
        }
        .login-container {
            background-color: #ffffff; /* White background for login */
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            margin: auto;
            width: 50%;
            text-align: center;
        }
        .button {
            background-color: #2ed573; /* Green button */
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            margin-top: 10px;
        }
        .button:hover {
            background-color: #1e7e36; /* Darker green on hover */
        }
        .checkbox {
            margin-bottom: 20px;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            font-size: 14px;
            color: #576574;
        }
        h3 {
            color: #2d3436;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<h1 class='title-text'>Welcome to Virtual Air Canvas</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Control your canvas with gestures and have fun drawing!</p>",
            unsafe_allow_html=True)

# Simulated database of users
users_db = {
    "admin": "1234"  # Username and password set to admin and 1234
}

# Initialize session state for login
if "login" not in st.session_state:
    st.session_state["login"] = False

# Login Section
if not st.session_state["login"]:
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)
    st.subheader("Login")
    loginname = st.text_input("Username")
    pw = st.text_input("Password", type="password")

    if st.button("Login", key="login_btn", help="Click to login"):
        if loginname in users_db and pw == users_db[loginname]:
            st.session_state["login"] = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password")

    st.markdown("</div>", unsafe_allow_html=True)  # Close login container
else:
    # Setup the layout with columns
    col1, col2 = st.columns([3, 2])

    with col1:
        start_drawing = st.checkbox("Start Drawing", value=False, help="Toggle drawing mode")
        FRAME_WINDOW = st.image([])  # Placeholder for image

    with col2:
        st.markdown("<h3>Drawing Options</h3>", unsafe_allow_html=True)
        color = st.radio("Choose your Color:", ('Blue', 'Green', 'Red', 'Yellow'))
        brush_size = st.slider("Brush Size:", min_value=5, max_value=50, value=10)  # Brush size slider
        clear_canvas = st.button("Clear Canvas", key="clear_canvas", help="Clear the drawing canvas")
        undo_action = st.button("Undo", key="undo_action", help="Undo last action")
        redo_action = st.button("Redo", key="redo_action", help="Redo last action")
        save_canvas = st.button("Save Canvas", key="save_canvas", help="Save your drawing")

    # Initialize drawing variables
    if "canvas" not in st.session_state:
        st.session_state["canvas"] = np.ones((480, 640, 3), dtype=np.uint8) * 255  # Create blank canvas
        st.session_state["history"] = []  # History for undo
        st.session_state["redo_stack"] = []  # Stack for redo
        st.session_state["last_point"] = None  # To store the previous position of the finger for smooth drawing

    canvas = st.session_state["canvas"]
    history = st.session_state["history"]
    redo_stack = st.session_state["redo_stack"]
    last_point = st.session_state["last_point"]

    # Color mapping
    color_map = {'Blue': (255, 0, 0), 'Green': (0, 255, 0), 'Red': (0, 0, 255), 'Yellow': (0, 255, 255)}

    # Clear the canvas
    if clear_canvas:
        canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
        history.clear()
        redo_stack.clear()
        last_point = None  # Reset last point
        st.session_state["canvas"] = canvas

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Start webcam feed only when the checkbox is checked
    if start_drawing:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Mirror the frame
            img = frame.copy()

            # Convert the image to RGB and process it with MediaPipe Hands
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_img)

            # Check if any hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the coordinates of the index finger tip (landmark 8) and thumb (landmark 4)
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                    h, w, _ = frame.shape
                    index_tip_coord = (int(index_tip.x * w), int(index_tip.y * h))
                    thumb_tip_coord = (int(thumb_tip.x * w), int(thumb_tip.y * h))

                    # Calculate the distance between the index finger and thumb tips
                    distance = np.linalg.norm(np.array(index_tip_coord) - np.array(thumb_tip_coord))

                    # Draw only when the index finger is detected and is open
                    if last_point and distance > 50:  # Check if the thumb and index finger are not close
                        cv2.line(canvas, last_point, index_tip_coord, color_map[color], brush_size)
                        st.session_state["history"].append(canvas.copy())  # Save state to history for undo
                    last_point = index_tip_coord if distance > 50 else None  # Update last_point

                    # Draw hand landmarks for visualization (optional)
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            else:
                last_point = None  # Reset last point if no hand is detected

            # Combine the frame and canvas
            image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)

            # Display the combined image in Streamlit
            FRAME_WINDOW.image(image_combined, channels="BGR")

            # Undo functionality
            if undo_action and history:
                redo_stack.append(canvas.copy())  # Save the current state to redo stack
                if len(history) > 1:
                    canvas = history.pop()  # Revert to the last state
                else:
                    canvas = history[0].copy()
                st.session_state["canvas"] = canvas

            # Redo functionality
            if redo_action and redo_stack:
                history.append(canvas.copy())  # Save the current state to history stack
                canvas = redo_stack.pop()  # Restore the last undone state
                st.session_state["canvas"] = canvas

            # Save the canvas functionality
            if save_canvas:
                cv2.imwrite("drawing.png", canvas)  # Save the drawing to a file
                st.success("Canvas saved as drawing.png")

        cap.release()

    else:
        st.warning("Check the box to start drawing!")

# Footer Section
st.markdown("<div class='footer'>Developed by our team</div>", unsafe_allow_html=True)








