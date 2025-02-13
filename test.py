import face_recognition
import cv2
import numpy as np
import os
import pickle
from PIL import Image, ImageDraw, ImageFont
from concurrent.futures import ThreadPoolExecutor
import time

# Путь к папке с базой лиц и к файлу для сохранения данных
base_dir = "images"
pupil_dir = os.path.join(base_dir, "pupil")
adult_dir = os.path.join(base_dir, "adult")
encodings_file = "face_encodings.pkl"

# Инициализация переменных
known_face_encodings = []
known_face_names = []
categories = []


def load_faces_and_save():
    """Загрузка лиц из базы данных и сохранение кодировок."""
    for directory, category in [(pupil_dir, "Ученик"), (adult_dir, "Взрослый")]:
        for file_name in os.listdir(directory):
            if file_name.endswith(('.jpg', '.jpeg', '.png')):  # проверяем, что файл изображение
                image_path = os.path.join(directory, file_name)
                person_name = os.path.splitext(file_name)[0]

                # Загрузка изображения и кодировка лица
                image = face_recognition.load_image_file(image_path)
                try:
                    face_encoding = face_recognition.face_encodings(image)[0]
                    known_face_encodings.append(face_encoding)
                    known_face_names.append(person_name)
                    categories.append(category)
                except IndexError:
                    print(f"Лицо не найдено на фото: {file_name}")

    # Сохранение кодировок в файл
    with open(encodings_file, "wb") as file:
        pickle.dump({
            "encodings": known_face_encodings,
            "names": known_face_names,
            "categories": categories,
        }, file)
    print("Кодировки лиц сохранены.")


def load_encodings():
    """Загрузка кодировок лиц из файла."""
    global known_face_encodings, known_face_names, categories
    with open(encodings_file, "rb") as file:
        data = pickle.load(file)
        known_face_encodings = data["encodings"]
        known_face_names = data["names"]
        categories = data["categories"]
    print("Кодировки лиц загружены.")


# Проверка наличия файла с кодировками
if os.path.exists(encodings_file):
    load_encodings()
else:
    load_faces_and_save()


# Функция для вывода текста с поддержкой Unicode
def draw_text(img, text, position, font_path="arial.ttf", font_size=20, color=(0, 255, 0)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# Инициализация камеры
video_capture = cv2.VideoCapture(0)


# Функция для обработки каждого кадра
def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Сжимаем изображение в 2 раза перед отправкой
    small_frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

    # Найти лица
    face_locations = face_recognition.face_locations(small_frame)

    if len(face_locations) > 0:
        try:
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        except IndexError:
            print("Не удалось получить кодировки лиц.")
            face_encodings = []
    else:
        face_encodings = []

    result = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Не найдено"
        category = ""

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                category = categories[best_match_index]

        # Если лицо не найдено, устанавливаем синюю рамку и синий текст
        if name == "Не найдено":
            color = (0, 0, 255)  # Синяя рамка
            text_color = (255, 0, 0)  # Синий текст
            label = "Не распознан"
        else:
            # Для учеников - зеленая рамка и зеленый текст
            if category == "Ученик":
                color = (0, 255, 0)  # Зеленая рамка
                text_color = (0, 255, 0)  # Зеленый текст
            # Для взрослых - красная рамка и красный текст
            else:
                color = (255, 0, 0)  # Красная рамка
                text_color = (0, 0, 255)  # Красный текст
            label = f"{category}: {name}"

        result.append(((left * 2, top * 2), (right * 2, bottom * 2), color, label, text_color))

    return frame, result


# Использование многозадачности для обработки кадров
with ThreadPoolExecutor() as executor:
    while True:
        ret, frame = video_capture.read()
        if ret:
            future = executor.submit(process_frame, frame)
            frame, results = future.result()

            # Рисуем результат на кадре
            for (left_top, right_bottom, color, label, text_color) in results:
                cv2.rectangle(frame, left_top, right_bottom, color, 2)
                frame = draw_text(frame, label, (left_top[0] + 6, right_bottom[1] - 25),
                                  font_path="C:/Windows/Fonts/arial.ttf", font_size=20, color=text_color)

            cv2.imshow('Видео', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

video_capture.release()
cv2.destroyAllWindows()
