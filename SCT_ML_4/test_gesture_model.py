import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# ✅ Load the trained model
print("Loading trained model...")
model = tf.keras.models.load_model("hand_gesture_model.h5")
print("✅ Model loaded successfully!")

# ✅ Enter the full path of any image you want to test
# Example: one image from your dataset folder
img_path = input("👉 Enter the full path of a gesture image to test: ")

# ✅ Load and preprocess the image
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # normalize pixel values

# ✅ Predict
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

# ✅ Your gesture labels (you can rename them if you know the exact gestures)
gesture_labels = ['Palm', 'L', 'Fist', 'Fist_moved', 'Thumb', 
                  'Index', 'OK', 'Palm_moved', 'C', 'Down']

print("\n✅ Prediction Result:")
print(f"Predicted gesture index: {predicted_class}")
print(f"Predicted gesture label: {gesture_labels[predicted_class]}")
