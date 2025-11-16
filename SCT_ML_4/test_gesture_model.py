import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
print("Loading trained model...")
model = tf.keras.models.load_model("hand_gesture_model.h5")
print("✅ Model loaded successfully!")
img_path = input("👉 Enter the full path of a gesture image to test: ")
img = image.load_img(img_path, target_size=(64, 64))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0 
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
gesture_labels = ['Palm', 'L', 'Fist', 'Fist_moved', 'Thumb', 
                  'Index', 'OK', 'Palm_moved', 'C', 'Down']
print("\n✅ Prediction Result:")
print(f"Predicted gesture index: {predicted_class}")
print(f"Predicted gesture label: {gesture_labels[predicted_class]}")
