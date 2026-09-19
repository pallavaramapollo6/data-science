import tkinter as tk
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# --------------------------------------------------
# 1. Load the trained MNIST model
# --------------------------------------------------

model = tf.keras.models.load_model("mnist_model.keras")


# --------------------------------------------------
# 2. Create the drawing window
# --------------------------------------------------

WINDOW_SIZE = 280

root = tk.Tk()
root.title("AI Handwritten Digit Recognizer")

canvas = tk.Canvas(
    root,
    width=WINDOW_SIZE,
    height=WINDOW_SIZE,
    bg="black"
)
canvas.pack(pady=10)


# PIL image used to store the drawing
image = Image.new("L", (WINDOW_SIZE, WINDOW_SIZE), 0)
draw = ImageDraw.Draw(image)


# --------------------------------------------------
# 3. Draw with the mouse
# --------------------------------------------------

def draw_digit(event):
    x = event.x
    y = event.y

    brush_size = 20

    canvas.create_oval(
        x - brush_size // 2,
        y - brush_size // 2,
        x + brush_size // 2,
        y + brush_size // 2,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        [
            x - brush_size // 2,
            y - brush_size // 2,
            x + brush_size // 2,
            y + brush_size // 2
        ],
        fill=255
    )


canvas.bind("<B1-Motion>", draw_digit)


# --------------------------------------------------
# 4. Ask AI to predict the digit
# --------------------------------------------------

def predict_digit():

    # Resize to MNIST's 28 x 28 pixels
    img = image.resize((28, 28))

    # Convert to NumPy array
    img_array = np.array(img)

    # Normalize pixel values
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = img_array.reshape(1, 28, 28)

    # AI prediction
    prediction = model.predict(img_array, verbose=0)

    digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    result_label.config(
        text=f"AI Prediction: {digit}\nConfidence: {confidence:.2f}%"
    )


# --------------------------------------------------
# 5. Clear the drawing
# --------------------------------------------------

def clear_canvas():

    canvas.delete("all")

    draw.rectangle(
        [0, 0, WINDOW_SIZE, WINDOW_SIZE],
        fill=0
    )

    result_label.config(
        text="Draw a digit and click Predict"
    )


# --------------------------------------------------
# 6. Buttons and result
# --------------------------------------------------

predict_button = tk.Button(
    root,
    text="Predict",
    command=predict_digit,
    font=("Arial", 14)
)

predict_button.pack(pady=5)


clear_button = tk.Button(
    root,
    text="Clear",
    command=clear_canvas,
    font=("Arial", 14)
)

clear_button.pack(pady=5)


result_label = tk.Label(
    root,
    text="Draw a digit and click Predict",
    font=("Arial", 16)
)

result_label.pack(pady=10)


# --------------------------------------------------
# Start application
# --------------------------------------------------

root.mainloop()
