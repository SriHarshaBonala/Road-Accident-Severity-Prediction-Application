import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.externals import joblib
app = Flask(__name__)

model = joblib.load('rf_clf_model.pkl')

list_dict = pickle.load(open('list_dict', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [int(x) for x in request.form.values()]
    Source = 0
    TMC = request.form["TMC"]
    Start_Lng = request.form["Start_Lng"]
    Start_Lat = request.form["Start_Lat"]
    Distance = request.form["Distance"]
    Side = list(list_dict[1].keys())[list(list_dict[1].values()).index(request.form["Side"])]
    City = list(list_dict[2].keys())[list(list_dict[2].values()).index(request.form["City"])]
    County = list(list_dict[3].keys())[list(list_dict[3].values()).index(request.form["County"])]
    State = list(list_dict[4].keys())[list(list_dict[4].values()).index(request.form["State"])]
    Timezone = list(list_dict[5].keys())[list(list_dict[5].values()).index(request.form["Timezone"])]
    Temperature = request.form["Temperature"]
    Humidity = request.form["Humidity"]
    Pressure = request.form["Pressure"]
    Visibility = request.form["Visibility"]
    Wind_Direction = list(list_dict[6].keys())[list(list_dict[6].values()).index(request.form["Wind_Direction"])]
    Weather_Condition = list(list_dict[7].keys())[list(list_dict[7].values()).index(request.form["Weather_Condition"])]
    Amenity = request.form["Amenity"]
    Bump = request.form["Bump"]
    Crossing = request.form["Crossing"]
    Give_Way = request.form["Give_Way"]
    Junction = request.form["Junction"]
    No_Exit = request.form["No_Exit"]
    Railway = request.form["Railway"]
    Roundabout = request.form["Roundabout"]
    Station = request.form["Station"]
    Stop = request.form["Stop"]
    Traffic_Calming = request.form["Traffic_Calming"]
    Traffic_Signal = request.form["Traffic_Signal"]
    Turning_Loop = request.form["Turning_Loop"]
    Sunrise_Sunset = list(list_dict[8].keys())[list(list_dict[8].values()).index(request.form["Sunrise_Sunset"])]
    Hour = request.form["Hour"]
    Weekday = list(list_dict[9].keys())[list(list_dict[9].values()).index(request.form["Weekday"])]
    Time_Duration = request.form["Time_Duration"]

    features = np.array([[Source, TMC, Start_Lng, Start_Lat, Distance, Side, City, County, State, Timezone, Temperature, Humidity, Pressure, Visibility, Wind_Direction, Weather_Condition, Amenity, Bump, Crossing, Give_Way, Junction, No_Exit, Railway, Roundabout, Station, Stop, Traffic_Calming, Traffic_Signal, Turning_Loop, Sunrise_Sunset, Hour, Weekday, Time_Duration]])


    output = model.predict(features)

    if(output==4):
        score='Very High!'
    elif(output==3):
        score='High!'
    elif(output==2):
        score='Medium'
    else:
        score='Low'

    return render_template('index.html', prediction_text='The Accident Severity is *{}* ,   Score : {} / 4'.format(score,output))

"""@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
"""

@app.route('/dashboard')
def dashboard():
    
    return render_template('dashboard.html')


if __name__ == "__main__":
    app.run(debug=True)