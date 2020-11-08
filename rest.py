from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np
import io
app = Flask(__name__)
costambedzie = None
 
values = {
 
'tens1': 0,
'tens2': 0,
'dist1': 0,
'dist2': 0
 
}

model = tf.keras.models.load_model('model.h5')

@app.route('/CZUJNIKI', methods = ['GET'])
def ZBIERZ_CZUJNIKI():
    return jsonify(values), 2
 
 
@app.route('/CZUJNIKI/<data>', methods = ['POST'])
def WYSLIJ_CZUJNIKI(data):
    result = data.split(',')
    result = [float(a.strip()) for a in result]
    values['tens1'] = result[0]
    values['tens2'] = result[1]
    values['dist1'] = result[2]
    values['dist2'] = result[3]
    return 200

@app.route('/obraz/', methods = ['POST'])
def obraz():
    img = request.get_json(force = True)['picture']
    img = np.asarray(Image.open(io.BytesIO(bytes(img.encode('ISO-8859-1'))))).reshape((1, 640, 480, 3))
    img = Image.fromarray(model()[0,...])
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue().decode('ISO-8859-1')
    
if __name__ == "__main__":
    app.run(port = 4200)