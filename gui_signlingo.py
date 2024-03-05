import tkinter as tk
from tkinter import ttk
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image, ImageTk

class SignLanguageRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Recognition App")

        self.video_label = ttk.Label(root)
        self.video_label.pack()

        self.letter_label = ttk.Label(root, text="Previous Letter: ")
        self.letter_label.pack()

        self.suggestions_label = ttk.Label(root, text="Suggestions: ")
        self.suggestions_label.pack()

        self.start_button = ttk.Button(root, text="Start Recognition", command=self.start_recognition)
        self.start_button.pack()

        self.stop_button = ttk.Button(root, text="Stop Recognition", command=self.stop_recognition)
        self.stop_button.pack()

        self.model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = self.model_dict['model']

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

        self.labels_dict = {0: 'A', 1: 'B', 2: 'L'}
        self.prev_letter = ''
        self.suggested_words = []
        self.recognition_active = False
        self.update()

    def start_recognition(self):
        self.recognition_active = True

    def stop_recognition(self):
        self.recognition_active = False

    def update(self):
        ret, frame = self.cap.read()
        if ret and self.recognition_active:
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        self.mp_hands.HAND_CONNECTIONS,  # hand connections
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = []
                    x_ = []
                    y_ = []

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10

                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10

                    prediction = self.model.predict([np.asarray(data_aux)])

                    predicted_character = self.labels_dict[int(prediction[0])]
                    self.prev_letter = predicted_character

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 480))
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.letter_label.config(text=f"Previous Letter: {self.prev_letter}")
        self.suggestions_label.config(text=f"Suggestions: {', '.join(self.suggested_words)}")

        self.root.after(10, self.update)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageRecognitionApp(root)
    app.run()
