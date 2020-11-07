from yolo_build import *

model = make_yolov3_model()
weigth_reader = WeightReader('ML\yolov3.weights')
weigth_reader.load_weights(model)
model.save('ML/model.h5')