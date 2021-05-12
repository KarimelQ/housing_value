from flask import Flask,render_template,request
app = Flask(__name__)
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('dataset.txt', delimiter = "|")

#@app.route('/')
#def hello_world():
#    scaler = prepare_data(df)
#    data = [[0,100,3,150],[0,150,5,200]]
#    data = scaler.transform(data).tolist()
#    data[0].extend([1,0,0,0])
#    data[1].extend([1,0,0,0])
#    model = load('model.joblib')
#    predictions = model.predict(data)
#    return str(predictions)


@app.route('/',methods=['GET','POST'])
def hello_world():
    request_type = request.method
    if(request_type=='GET'):
        return render_template('index.html',href = 'static/bordeaux.jpg')
    else:
        scaler = prepare_data(df)
        text = request.form['text']
        list_data = float_str_to_np_array(text)
        data = scaler.transform([list_data,list_data]).tolist()
        data[0].extend([1,0,0,0])
        data[1].extend([1,0,0,0])
        model = load('model.joblib')
        predictions = int(model.predict(data)[0])

        return str(predictions)


def float_str_to_np_array(floats_str):
  def is_float(s):
    try:
      float(s)
      return True
    
    except:
      return False
  floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
  
  return floats

#### Nombre de lots   --      Surface reelle bati   --    Nombre pieces principales   --   Surface terrain
def prepare_data(df):
    scaler = MinMaxScaler()
    scaler = scaler.fit(df)
    #scaler = scaler.transform(df)
    return scaler
