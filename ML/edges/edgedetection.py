from PIL import Image
import numpy as np
import tensorflow as tf
import coremltools as ct
import coremltools.proto.FeatureTypes_pb2 as ft 
import matplotlib.pyplot as plt

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

def create_model(IMG_SHAPE = (480, 640)):
    sobel_x = tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
    filter = tf.concat([sobel_x_filter, sobel_y_filter], -1)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape = (IMG_SHAPE[0], IMG_SHAPE[1], 3)))
    model.add(tf.keras.layers.Conv2D(1, (1, 1), (1, 1), padding = 'valid', use_bias = False))
    model.add(tf.keras.layers.Conv2D(2, (3, 3), (1, 1), padding = 'same', use_bias = False))
    model.add(tf.keras.layers.Conv2D(1, (1, 1), (1, 1), padding = 'same', use_bias = False))
    model.add(tf.keras.layers.Conv2D(3, (1, 1), (1, 1), padding = 'same', use_bias = False, name = 'output'))
    #model.add(tf.keras.layers.Flatten())
    #model.add(tf.keras.layers.Lambda(lambda x: abs(x), name = 'output'))

    HALFAWAREOWER = np.ones((1,3))/3
    x = tf.convert_to_tensor(model.layers[0].get_weights())
    x = np.array(x)
    x[0,...,0] = np.expand_dims(HALFAWAREOWER,1)
    model.layers[0].set_weights(x)

    x = tf.convert_to_tensor(model.layers[1].get_weights())
    x = np.array(x)
    x[0,...,0] = np.expand_dims(sobel_x,-1)
    x[0,...,1] = np.expand_dims(np.array(sobel_x).T,-1)
    model.layers[1].set_weights(x)

    HALFAWAREOWER = np.ones((1,2))
    x = tf.convert_to_tensor(model.layers[2].get_weights())
    x = np.array(x)
    x[0,...,0] = np.expand_dims(HALFAWAREOWER,1)
    model.layers[2].set_weights(x)

    model.layers[3].set_weights(np.array([0, 0.15, 0.4]).reshape(np.array(model.layers[3].get_weights()).shape)/30)

    return model

def save_model_as_coreml(model, IMG_SHAPE = (480, 640)):
    mlmodel = ct.convert(model)#, inputs= [ct.ImageType('Layer', color_layout='RGB', shape=(1, 3, IMG_SHAPE[0], IMG_SHAPE[1], 1))])
    mlmodel.save('ML/edges/edgedetection.mlmodel')

    spec = ct.utils.load_spec("ML/edges/edgedetection.mlmodel")
    inputs = spec.description.input[0]
    inputs.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    inputs.type.imageType.height = IMG_SHAPE[1]
    inputs.type.imageType.width = IMG_SHAPE[0]
    
    # output = spec.description.output[0]
    # output.type.imageType.colorSpace = ft.ImageFeatureType.RGB
    # output.type.imageType.height = IMG_SHAPE[1]
    # output.type.imageType.width = IMG_SHAPE[0]

    print(spec.description)
    ct.utils.save_spec(spec, "ML/edges/edgedetection.mlmodel")
    #
    # mlmodel.save('ML/edges/edgedetection.mlmodel')

model = create_model()
plt.imshow(model([np.asarray(Image.open(r'C:\Users\Krzysztof Kramarz\Desktop\hackathon spacehacks\Spacehack\ML\depth\test.png').resize((480, 640))).reshape((1, 640, 480, 3))])[0,...])
plt.show()
print(model.summary)
save_model_as_coreml(model)




