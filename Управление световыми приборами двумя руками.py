import cv2
import mediapipe as mp
from stupidArtnet import StupidArtnet  # <-- добавили

# ----- Настройки камеры / жестов -----
CAM_ID = 0  # индекс веб-камеры
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ----- Настройки DMX / Art-Net -----
TARGET_IP = "2.129.134.206"  # IP DiaLight AC Gate-8 (сейчас broadcast)
UNIVERSE = 0                   # попробуешь 0 или 1, смотря что у Gate-8
PACKET_SIZE = 512

# голова 1, адрес 001 → каналы 1–13 → индексы 0–12
HEAD_BASE = 0      # смещение первой головы (0 для адреса 1)
PAN_CH     = HEAD_BASE + 0  # CH1  – PAN
TILT_CH    = HEAD_BASE + 1  # CH3  – TILT
SPEED_CH   = HEAD_BASE + 13  # CH12  – PAN/TILT speed
DIMMER_CH  = HEAD_BASE + 3  # CH10  – Dimmer
SHUTTER_CH = HEAD_BASE + 8  # CH9  – Shutter / Strobe
RED_CH     = HEAD_BASE + 4  # CH5  – R
GREEN_CH   = HEAD_BASE + 5  # CH6  – G
BLUE_CH    = HEAD_BASE + 6  # CH7 – B
WHITE_CH   = HEAD_BASE + 7 # CH8 – W
# CH12, CH13 пока не трогаем

# чувствительность панорамирования/наклона к level_x/level_y
PAN_SENS  = 0.25  # 0.25 значит: level_x ~ ±500 → pan ~ 0..255
TILT_SENS = 0.25

# ----- Инициализация MediaPipe -----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Индексы кончиков пальцев
TIP_IDS = {
    "THUMB": 4,
    "INDEX": 8,
    "MIDDLE": 12,
    "RING": 16,
    "PINKY": 20
}


def fingers_up(hand_landmarks, handedness_label):
    """
    Возвращает число поднятых пальцев и список булевых значений [thumb, index, middle, ring, pinky]
    """
    landmark = hand_landmarks.landmark
    fingers = []

    # Большой палец — по X
    if handedness_label == "Right":
        thumb_is_up = not (
            landmark[TIP_IDS["THUMB"]].x > landmark[5].x
            and landmark[TIP_IDS["THUMB"]].x < landmark[17].x
        )
    else:  # "Left"
        thumb_is_up = not (
            landmark[TIP_IDS["THUMB"]].x < landmark[5].x
            and landmark[TIP_IDS["THUMB"]].x > landmark[17].x
        )
    fingers.append(1 if thumb_is_up else 0)

    # Остальные пальцы — по Y (tip.y < pip.y -> поднят)
    for id_name in ("INDEX", "MIDDLE", "RING", "PINKY"):
        tip_id = TIP_IDS[id_name]
        pip_id = tip_id - 2
        if landmark[tip_id].y < landmark[pip_id].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers), fingers


def distance(finger1, finger2, hand_landmarks):
    landmark = hand_landmarks.landmark
    dx = landmark[finger1].x - landmark[finger2].x
    dy = landmark[finger1].y - landmark[finger2].y
    return dx * dx + dy * dy


def main():
    # Текущие координаты пальца относительно камеры (необязательно хранить)
    coordinates_y = 0.0
    coordinates_x = 0.0

    # Текущее положение головы
    pan_var = 128
    tilt_var = 128

    # Базовая точка, от которой считаем смещение во время ЖЕСТА
    pan_null = 128
    tilt_null = 128
    start_x = 0.0
    start_y = 0.0

    # Флаг: сейчас жест активен (два пальца вместе)
    gesture_active = False

    # ----- Инициализируем камеру -----
    cap = cv2.VideoCapture(CAM_ID)
    if not cap.isOpened():
        print("Не удалось открыть камеру. Проверьте индекс камеры и подключение.")
        return

    # ----- Инициализируем Art-Net / DMX -----
    artnet = StupidArtnet(TARGET_IP, UNIVERSE, PACKET_SIZE)
    dmx = bytearray(PACKET_SIZE)

    # базовое состояние головы: скорость, диммер, шторка, белый цвет
    dmx[SPEED_CH]   = 20   # CH5: 010–255 fast→slow, 200 = достаточно плавно
    dmx[DIMMER_CH]  = 200  # CH6: 100 % яркость
    #dmx[SHUTTER_CH] = 0   # CH7: 250–255 = постоянно открыта (без строба)
    dmx[RED_CH]     = 0
    #dmx[GREEN_CH]   = 0
    dmx[BLUE_CH]    = 255
    dmx[WHITE_CH]   = 0  # белый

    dmx[11+17] = 0
    dmx[9+17] = 128
    dmx[8+17] = 0
    dmx[4+17] = 0
    dmx[5+17] = 0
    dmx[6+17] = 0
    dmx[7+17] = 255

    dmx[11+34] = 0
    dmx[9+34] = 200
    dmx[8+34] = 0
    dmx[4+34] = 0
    dmx[5+34] = 0
    dmx[6+34] = 255
    dmx[7+34] = 0

    dmx[SPEED_CH+51] = 20  # CH5: 010–255 fast→slow, 200 = достаточно плавно
    dmx[DIMMER_CH+51] = 128  # CH6: 100 % яркость
    # dmx[SHUTTER_CH] = 0   # CH7: 250–255 = постоянно открыта (без строба)
    dmx[RED_CH+51] = 255
    # dmx[GREEN_CH]   = 0
    dmx[BLUE_CH+51] = 0
    dmx[WHITE_CH+51] = 0  # белый

    head = 0

    print("Старт. Жест правой рукой (2 пальца вместе) управляет PAN (x) и TILT (y).")

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка чтения с камеры.")
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            counts = {"Left": None, "Right": None}

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label  # "Left" или "Right"
                    score = handedness.classification[0].score

                    count, fingers_list = fingers_up(hand_landmarks, label)
                    counts[label] = (count, fingers_list, score)

                    landmark = hand_landmarks.landmark

                    if label == "Right":
                        if landmark[6].y > landmark[8].y and landmark[10].y > landmark[12].y and distance(8, 12, hand_landmarks) < 0.005 and counts['Right'][0] == 2:
                            pinch_cond = True
                        else:
                            pinch_cond = False

                    # --- определяем условие ЖЕСТА ---

                    if label == "Left":
                        head = (counts['Left'][0] - 1)*17

                    if pinch_cond:
                        # коорды пальца
                        coordinates_y = landmark[8].y
                        coordinates_x = landmark[8].x

                        # если жест только что НАЧАЛСЯ — захватываем базу
                        if not gesture_active:
                            gesture_active = True
                            start_x = coordinates_x
                            start_y = coordinates_y
                            pan_null = pan_var      # текущая позиция головы -> базовая
                            tilt_null = tilt_var

                        # смещение пальца относительно стартовой точки (в жесте)
                        level_y = (coordinates_y - start_y) * 1000.0
                        level_x = (coordinates_x - start_x) * 1000.0

                        print("y:", level_y, " x:", level_x)

                        # вычисляем pan/tilt вокруг сохранённой базы
                        pan = int(pan_null + level_x * PAN_SENS)
                        tilt = int(tilt_null - level_y * TILT_SENS)

                        # ограничиваем 0..255
                        pan = max(0, min(255, pan))
                        tilt = max(0, min(255, tilt))

                        # пишем в DMX-кадр
                        if head >=0:
                            dmx[PAN_CH+head] = pan
                            dmx[TILT_CH+head] = tilt

                        if head == 68:
                            dmx[PAN_CH] = pan
                            dmx[TILT_CH] = tilt

                            dmx[17] = pan
                            dmx[19] = tilt

                            dmx[34] = pan
                            dmx[36] = tilt

                            dmx[PAN_CH+51] = pan
                            dmx[TILT_CH+51] = tilt


                        #dmx[PAN_CH+17] = pan
                        #dmx[TILT_CH+17] = tilt

                        #dmx[PAN_CH+34] = pan
                        #dmx[TILT_CH+34] = tilt
                        # сохраняем последнюю позицию как текущее положение головы
                        pan_var = pan
                        tilt_var = tilt
                        print("pan_var:", pan_var, "tilt_var:", tilt_var, head)

                    else:
                        # Жеста нет → просто считаем, что он закончился,
                        # НО pan_var / tilt_var / dmx НЕ трогаем,
                        # голова остаётся в последней позиции.
                        gesture_active = False

                    # отправляем кадр (даже если не изменился — не страшно)
                    artnet.set(dmx)
                    artnet.show()

                    # Рисуем скелет и точки
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

                    # Подпись над запястьем
                    h, w, _ = image.shape
                    wrist = hand_landmarks.landmark[0]
                    cx, cy = int(wrist.x * w), int(wrist.y * h)
                    text = f"{label} hand: {count}"
                    cv2.putText(image, text, (cx - 40, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Текст слева / справа
            left_text = f"Left: {counts['Left'][0] if counts['Left'] else 0}"
            right_text = f"Right: {counts['Right'][0] if counts['Right'] else 0}"
            cv2.putText(image, left_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(image, right_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow('Hands - finger count (press q to quit)', image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    try:
        artnet.stop()
    except Exception:
        pass


if __name__ == "__main__":
    main()
