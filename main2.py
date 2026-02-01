import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from statistics import median

#Анизоцитоз

def find_deviating_contours(contours):
    # Вычисляем длину каждого контура
    lengths = [cv2.arcLength(cnt, True) for cnt in contours]
    
    # Вычисляем квартильные значения и межквартильный размах (IQR)
    q1 = np.percentile(lengths, 25)
    q3 = np.percentile(lengths, 75)
    iqr = q3 - q1
    lower_threshold = q1 - 1.5 * iqr
    upper_threshold = q3 + 1.5 * iqr

    print(f"Q1 = {q1}, Median = {median(lengths)}, Q3 = {q3}, IQR = {iqr}")
    print(f"Порог снизу: {lower_threshold}, порог сверху: {upper_threshold}")

    # Разделяем контуры на уменьшённые и увеличенные
    small_contours = []
    large_contours = []
    small_lengths = []
    large_lengths = []
    for cnt, l in zip(contours, lengths):
        if l < lower_threshold:
            small_contours.append(cnt)
            small_lengths.append(l)
        elif l > upper_threshold:
            large_contours.append(cnt)
            large_lengths.append(l)
    return small_contours, large_contours, small_lengths, large_lengths

def process_image(image_path):
    global lengths_label
    # Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        print("Не удалось загрузить изображение!")
        return

    original_img = img.copy()

    # 1. Детекция контуров через пороговую обработку
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours_tree, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Рисуем найденные контуры (пропуская первый, как в исходном коде)
    for i, cnt in enumerate(contours_tree):
        if i == 0:
            continue
        cv2.drawContours(img, [cnt], -1, (0, 255, 0), 2)
    cv2.imshow('Shapes', img)

    # 2. Детекция краёв и повторное нахождение контуров
    gray2 = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray2, 50, 200)
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print("Контуры не найдены!")
        return

    # Сортируем контуры по площади (от большего к меньшему)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(sorted_contours) < 2:
        print("Найдено недостаточно контуров для анализа!")
        return

    # Получаем уменьшённые и увеличенные контуры и их длины
    small_contours, large_contours, small_lengths, large_lengths = find_deviating_contours(sorted_contours)
    total_deviations = len(small_contours) + len(large_contours)
    
    # Формируем строку с информацией
    lengths_str = (f"Уменьшенные клетки (кол-во: {len(small_contours)}): " +
                   ' '.join(map(str, small_lengths)) + "\n" +
                   f"Увеличенные клетки (кол-во: {len(large_contours)}): " +
                   ' '.join(map(str, large_lengths)) + "\n" +
                   f"Всего отклоняющихся: {total_deviations}")
    print(lengths_str)
    
    # Обновляем метку в главном окне
    lengths_label.config(text=lengths_str)
    root.update()
    
    # Рисуем уменьшённые контуры синим, увеличенные — красным
    for cnt in small_contours:
        cv2.drawContours(original_img, [cnt], -1, (255, 0, 0), 2)  # синий для уменьшенных
    for cnt in large_contours:
        cv2.drawContours(original_img, [cnt], -1, (0, 255, 0), 2)  # зеленый для увеличенных
    
    cv2.imshow('contours', original_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def open_image():
    # Открываем диалог выбора файла
    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif")])
    if file_path:
        process_image(file_path)

if __name__ == '__main__':
    # Создаем главное окно GUI
    root = tk.Tk()
    root.title("Нахождение отклоняющихся клеток")
    root.geometry("400x250")

    # Кнопка для загрузки изображения
    btn_load = tk.Button(root, text="Загрузить изображение", command=open_image)
    btn_load.pack(pady=10)

    # Метка для вывода информации по клеткам
    lengths_label = tk.Label(root, text="Информация об отклонениях клеток:", wraplength=380, justify="left")
    lengths_label.pack(pady=10)

    root.mainloop()
