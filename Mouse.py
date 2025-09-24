import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()
cap_w, cap_h = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, cap_w)
cap.set(4, cap_h)

prev_x, prev_y = 0, 0
smooth_factor = 0.5
drag_smooth_factor = 0.2
tap_threshold = 0.03
pinching = False
recording = False
paused = False
sequence = []

combo_actions = {
    (("01000","01000"),("11111","01000")): lambda: print("Combo 1 triggered!"),
    (("11111","00000"),("01100","00000")): lambda: print("Combo 2 triggered!")
}

def finger_up_pattern(lm):
    tips = [4,8,12,16,20]
    pips = [3,6,10,14,18]
    pattern = ""
    for tip,pip in zip(tips,pips):
        pattern += "1" if lm.landmark[tip].y < lm.landmark[pip].y else "0"
    return pattern

def hand_to_screen(x, y):
    screen_x = int(x * screen_w)
    screen_y = int(y * screen_h)
    screen_x = max(0, min(screen_w-1, screen_x))
    screen_y = max(0, min(screen_h-1, screen_y))
    return screen_x, screen_y

with mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        left_pattern = right_pattern = "00000"
        left_hand = right_hand = None
        drag_status = False
        click_status = False
        zoom_status = ""
        combo_text = ""

        if result.multi_hand_landmarks and result.multi_handedness:
            for lm, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                label = handedness.classification[0].label
                pattern = finger_up_pattern(lm)
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                if label == "Right":
                    right_pattern = pattern
                    right_hand = lm
                else:
                    left_pattern = pattern
                    left_hand = lm

        if not paused:
            # --- Right-hand cursor + drag/click ---
            if right_hand:
                ix, iy = right_hand.landmark[8].x, right_hand.landmark[8].y
                cursor_x, cursor_y = hand_to_screen(ix, iy)

                # Drag (smooth) using index+middle fingers
                if right_pattern[1] == "1" and right_pattern[2] == "1":
                    pyautogui.mouseDown()
                    cursor_x = int(prev_x + (cursor_x - prev_x) * drag_smooth_factor)
                    cursor_y = int(prev_y + (cursor_y - prev_y) * drag_smooth_factor)
                    pyautogui.moveTo(cursor_x, cursor_y, duration=0.01)
                    drag_status = True
                else:
                    pyautogui.mouseUp()

                # Click (middle finger + thumb)
                mx, my = right_hand.landmark[12].x, right_hand.landmark[12].y
                tx, ty = right_hand.landmark[4].x, right_hand.landmark[4].y
                pinch_dist = ((mx - tx)**2 + (my - ty)**2)**0.5

                if pinch_dist < tap_threshold:
                    if not pinching:
                        pinching = True
                        click_status = True
                else:
                    if pinching:
                        if not drag_status:
                            pyautogui.click()
                        pinching = False
                        click_status = False

                # Smooth movement when not dragging
                if not drag_status:
                    cursor_x = int(prev_x + (cursor_x - prev_x) * smooth_factor)
                    cursor_y = int(prev_y + (cursor_y - prev_y) * smooth_factor)
                    pyautogui.moveTo(cursor_x, cursor_y, duration=0.01)

                prev_x, prev_y = cursor_x, cursor_y

            # --- Left-hand zoom ---
            if left_hand:
                if left_pattern[1] == "1" and left_pattern[2] == "0":
                    pyautogui.scroll(20)
                    zoom_status = "ZOOM IN"
                elif left_pattern[1] == "1" and left_pattern[2] == "1":
                    pyautogui.scroll(-20)
                    zoom_status = "ZOOM OUT"

            # --- Combo gestures ---
            both_pattern = (left_pattern, right_pattern)
            both_open = left_pattern == "11111" and right_pattern == "11111"
            both_closed = left_pattern == "00000" and right_pattern == "00000"

            if both_open and not recording:
                recording = True
                sequence = []
                combo_text = "Combo Recording: ON"
            elif both_closed and recording:
                recording = False
                combo_text = "Combo Recording: OFF"
                print("Combo sequence:", sequence)
                action = combo_actions.get(tuple(sequence))
                if action:
                    print("Triggered combo!")
                    action()
                sequence = []
            elif recording:
                if not sequence or sequence[-1] != both_pattern:
                    sequence.append(both_pattern)
                    combo_text = f"Combo Step: {len(sequence)}"

        else:
            combo_text = "PAUSED"

        # --- Overlay ---
        y = 30
        if drag_status: cv2.putText(frame, "DRAG", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2); y+=30
        if click_status: cv2.putText(frame, "CLICK", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2); y+=30
        if zoom_status: cv2.putText(frame, zoom_status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2); y+=30
        if combo_text: cv2.putText(frame, combo_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2); y+=30

        # Cursor circle
        cv2.circle(frame, (prev_x, prev_y), 10, (255,0,0), 2)

        cv2.imshow("Virtual Mousepad Overlay", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('*'):  # pause toggle
            paused = not paused

cap.release()
cv2.destroyAllWindows()
