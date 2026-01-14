import cv2
import mediapipe as mp
import numpy as np
import time
import random


class AirDrawing:
    def __init__(self):
        # Mediapipe setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

        # Canvas
        self.canvas = None
        self.brush_size = 8
        self.opacity = 0.6

        # Colors (palette)
        self.colors = [
            (0, 0, 255),     # Red
            (0, 255, 0),     # Green
            (255, 0, 0),     # Blue
            (0, 255, 255),   # Yellow
            (255, 0, 255),   # Magenta
            (255, 255, 255)  # White
        ]
        self.color_index = 0
        self.current_color = self.colors[self.color_index]

        # Gesture states
        self.prev_x, self.prev_y = None, None
        self.last_gesture_time = 0
        self.cooldown = 1.0  # sec between gesture detections

    def fingers_up(self, landmarks, h, w):
        """Detect which fingers are up"""
        finger_tips = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if landmarks[finger_tips[0]].x < landmarks[finger_tips[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for tip in finger_tips[1:]:
            if landmarks[tip].y < landmarks[tip - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers  # 1 = up, 0 = down

    def draw_palette(self, img):
        """Draw color palette at top"""
        x, y, box_size = 20, 20, 40
        for i, color in enumerate(self.colors):
            cv2.rectangle(img, (x + i * (box_size + 10), y),
                          (x + i * (box_size + 10) + box_size, y + box_size),
                          color, -1)
            if i == self.color_index:
                cv2.rectangle(img, (x + i * (box_size + 10), y),
                              (x + i * (box_size + 10) + box_size, y + box_size),
                              (255, 255, 255), 2)

        cv2.putText(img, "Press S to Save", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            if self.canvas is None:
                self.canvas = np.zeros((h, w, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    lm = hand_landmarks.landmark
                    fingers = self.fingers_up(lm, h, w)

                    cx, cy = int(lm[8].x * w), int(lm[8].y * h)  # index tip

                    now = time.time()

                    # ✍️ Drawing mode (only index finger up)
                    if fingers[1] == 1 and sum(fingers) == 1:
                        if self.prev_x is None:
                            self.prev_x, self.prev_y = cx, cy
                        cv2.line(self.canvas, (self.prev_x, self.prev_y), (cx, cy),
                                 self.current_color, self.brush_size)
                        self.prev_x, self.prev_y = cx, cy
                    else:
                        self.prev_x, self.prev_y = None, None

                    # 🖐️ Palm swipe right (clear screen)
                    if all(fingers):  # all fingers up
                        if now - self.last_gesture_time > self.cooldown:
                            self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
                            self.last_gesture_time = now

                    # ✌️ Peace sign = change color
                    if fingers[1] == 1 and fingers[2] == 1 and sum(fingers) == 2:
                        if now - self.last_gesture_time > self.cooldown:
                            self.color_index = (self.color_index + 1) % len(self.colors)
                            self.current_color = self.colors[self.color_index]
                            self.last_gesture_time = now

            # Overlay drawing
            combined = cv2.addWeighted(frame, 1 - self.opacity, self.canvas, self.opacity, 0)

            # Draw palette
            self.draw_palette(combined)

            cv2.imshow("Air Drawing", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                filename = f"drawing_{random.randint(1000,9999)}.png"
                cv2.imwrite(filename, self.canvas)
                print(f"✅ Saved as {filename}")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = AirDrawing()
    app.run()
