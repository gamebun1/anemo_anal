import tkinter as tk
from tkinter import filedialog
import cv2
from statistics import median

def diff_finder(contours):
    # Вычисляем длину каждого контура
    length_arr = [cv2.arcLength(cnt, True) for cnt in contours]
    med_val = median(length_arr)
    print(f"Median длина: {med_val}")
    # Выбираем те контуры, у которых длина больше медианы плюс 20
    diff_contours = []
    big_lengths = []
    for cnt, l in zip(contours, length_arr):
        if l > med_val + 20:
            diff_contours.append(cnt)
            big_lengths.append(l)
    return diff_contours, big_lengths

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
        print("Найдено недостаточно контуров для выделения двух самых больших!")
        return

    # Получаем отличающиеся контуры и их длины
    diff_contours, big_lengths = diff_finder(sorted_contours)
    lengths_str = f"Длины отличающихся контуров: {' '.join(map(str, big_lengths))}"
    print(lengths_str)
    # Обновляем метку в главном окне
    lengths_label.config(text=lengths_str)
    root.update()
    
    # Рисуем все отличающиеся контуры на оригинальном изображении
    for cnt in diff_contours:
        cv2.drawContours(original_img, [cnt], -1, (0, 0, 0), 2)
    
    cv2.imshow('Largest Contours', original_img)
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
    root.title("Нахождение Контуров")
    root.geometry("400x200")

    # Кнопка для загрузки изображения
    btn_load = tk.Button(root, text="Загрузить изображение", command=open_image)
    btn_load.pack(pady=10)

    # Метка для вывода длин отличающихся контуров
    lengths_label = tk.Label(root, text="Длины отличающихся контуров: ", wraplength=380, justify="left")
    lengths_label.pack(pady=10)

    root.mainloop()
