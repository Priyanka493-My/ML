import tensorflow as tf
import tf_keras

# Load the legacy .h5 model using tf_keras
model = tf_keras.models.load_model('clothing-model.h5')

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('clothing-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)

print("Successfully generated clothing-model.tflite!")