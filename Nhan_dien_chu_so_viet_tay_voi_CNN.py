import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

GRID_SIZE = 28
PIXEL_SIZE = 20

pixel_matrix = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

model = tf.keras.models.load_model("model.h5")


def preprocess_input(matrix):
    array = np.array(matrix, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    array = np.expand_dims(array, axis=-1)
    return array


def predict_and_display():
    input_data = preprocess_input(pixel_matrix)
    predictions = model.predict(input_data)[0]

    for i, prob in enumerate(predictions):
        gray_value = int((1 - prob) * 255)
        color = f"#{gray_value:02x}{gray_value:02x}{gray_value:02x}"

        canvas.create_oval(
            GRID_SIZE * PIXEL_SIZE + 40, i * 50 + 30,
            GRID_SIZE * PIXEL_SIZE + 80, i * 50 + 70,
            fill=color, outline=color
        )

        canvas.create_text(
            GRID_SIZE * PIXEL_SIZE + 90, i * 50 + 50,
            text=str(i), font=("Helvetica", 12), fill="black"
        )

    predicted_label = np.argmax(predictions)
    predicted_prob = predictions[predicted_label]

    canvas.delete("prediction_label")
    canvas.create_text(
        GRID_SIZE * PIXEL_SIZE + 40, 10 * 50 + 30,
        text=f"{predicted_label} ({predicted_prob * 100:.2f}%)",
        font=("Helvetica", 10), fill="black", tags="prediction_label"
    )


def draw_pixel(event):
    x, y = event.x // PIXEL_SIZE, event.y // PIXEL_SIZE
    intensity = 255
    weight_matrix = [
        [0.2, 0.658, 0.0],
        [0.658, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
        for i in range(-1, 2):
            for j in range(-1, 2):
                nx, ny = x + i, y + j
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    new_intensity = int(intensity * weight_matrix[i + 1][j + 1])
                    pixel_matrix[ny][nx] = max(pixel_matrix[ny][nx], new_intensity)
                    color = f"#{pixel_matrix[ny][nx]:02x}{pixel_matrix[ny][nx]:02x}{pixel_matrix[ny][nx]:02x}"
                    canvas.create_rectangle(
                        nx * PIXEL_SIZE, ny * PIXEL_SIZE,
                        (nx + 1) * PIXEL_SIZE, (ny + 1) * PIXEL_SIZE,
                        fill=color, outline=""
                    )

        predict_and_display()


def clear_canvas():
    canvas.delete("all")
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            pixel_matrix[i][j] = 0
            canvas.create_rectangle(
                j * PIXEL_SIZE, i * PIXEL_SIZE,
                (j + 1) * PIXEL_SIZE, (i + 1) * PIXEL_SIZE,
                fill="black", outline=""
            )

    for i in range(10):
        canvas.create_oval(
            GRID_SIZE * PIXEL_SIZE + 40, i * 50 + 30,
            GRID_SIZE * PIXEL_SIZE + 80, i * 50 + 70,
            fill="white", outline="white"
        )
        canvas.create_text(
            GRID_SIZE * PIXEL_SIZE + 90, i * 50 + 50,
            text=str(i), font=("Helvetica", 12), fill="black"
        )





def save_as_png():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
    )
    if file_path:
        img = Image.new("L", (GRID_SIZE, GRID_SIZE))
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                img.putpixel((j, i), pixel_matrix[i][j])
        img.save(file_path)


def import_image():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
    )
    if file_path:
        img = Image.open(file_path).convert("L")
        img = img.resize((GRID_SIZE, GRID_SIZE))
        img_data = np.array(img)

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                pixel_matrix[i][j] = img_data[i, j]
                color = f"#{img_data[i, j]:02x}{img_data[i, j]:02x}{img_data[i, j]:02x}"
                canvas.create_rectangle(
                    j * PIXEL_SIZE, i * PIXEL_SIZE,
                    (j + 1) * PIXEL_SIZE, (i + 1) * PIXEL_SIZE,
                    fill=color, outline=""
                )

        predict_and_display()


root = tk.Tk()
root.title("Nhận diện chữ số viết tay với CNN")
root.resizable(False, False)

canvas = tk.Canvas(root, width=GRID_SIZE * PIXEL_SIZE + 120, height=GRID_SIZE * PIXEL_SIZE, bg="white")
canvas.pack()

for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        canvas.create_rectangle(
            j * PIXEL_SIZE, i * PIXEL_SIZE,
            (j + 1) * PIXEL_SIZE, (i + 1) * PIXEL_SIZE,
            fill="black", outline=""
        )

clear_button = tk.Button(root, text="Xóa bảng", command=clear_canvas)
clear_button.pack()

save_button = tk.Button(root, text="Lưu dưới dạng PNG", command=save_as_png)
save_button.pack()

import_button = tk.Button(root, text="Nhập ảnh", command=import_image)
import_button.pack()

canvas.bind("<B1-Motion>", draw_pixel)
canvas.bind("<Button-1>", draw_pixel)

root.mainloop()


root.mainloop()