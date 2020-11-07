# from keras.preprocessing.image import load_img, img_to_array
# from keras.models import load_model
import numpy as np
import tensorflow as tf
from PIL import Image

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    print('dupa')
    pass

yolo_model = tf.keras.models.load_model('model.h5')
yolo_model.summary()
image = Image.open('ML\\test.png').resize((416, 416))
image = np.asarray(image)
image = image.astype('float32')
image/=255.0
image = np.expand_dims(image, 0)
prediction = yolo_model(image)
image = Image.fromarray(prediction[0])
image.save('test_pred.png')