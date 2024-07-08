from flask import Flask, render_template, request, send_from_directory
import pickle
import pandas as pd
from catboost import CatBoostClassifier
import io
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt
import base64
import os



app = Flask(__name__)

@app.route("/")
def index():
    types = ["H" , "L" , "M"]
    return render_template("index.html" , types = types)

@app.route('/predict' , methods = ['POST'])

def predict():

    if request.method == "POST":
        
        if request.form["type"] == "H":
            type = 0.0
        
        if request.form["type"] == "L":
            type = 1.0
        
        if request.form["type"] == "M":
            type = 2.0

        air_temp = request.form["air_temp"]
        process_temp = request.form["process_temp"]
        rot_speed = request.form["rot_speed"]
        torque = request.form["torque"]
        tool = request.form["tool"]


        input_data = pd.DataFrame({
            'Type' : [float(type)],  
            'Air temperature [K]' : [float(air_temp)],
            'Process temperature [K]' : [float(process_temp)],
            'Rotational speed [rpm]' : [float(rot_speed)],
            'Torque [Nm]' : [float(torque)],
            'Tool wear [min]' : [float(tool)]
        })

        model = pickle.load(open("Pred_Maintainance_final_ .pkl","rb"))
        prediction = model.predict(input_data)[0]

        if prediction == 0:
            output = "No Failure"

        elif prediction == 1:
            output = "Heat Dissipation Failure"
        
        elif prediction == 2:
            output = "Power Failure"
        
        elif prediction == 3:
            output = "Overstrain Failure"

        elif prediction == 4:
            output = "Tool Wear Failure"

        elif prediction == 5:
            output = "Random Failures"


        X1_res = pd.read_csv("X1_res.csv")

        X1_res = X1_res.drop(["Unnamed: 0"] , axis = 1)
        
        explainer = lime_tabular.LimeTabularExplainer(
            X1_res.values,
            feature_names=X1_res.columns,
            class_names=model.classes_,
            mode='classification'
        )

        explanation = explainer.explain_instance(
            input_data.values[0],
            model.predict_proba,
            num_features=len(input_data.columns)
        )

        fig = explanation.as_pyplot_figure()
        lime_image_path = 'static/lime_explanation.png'
        fig.savefig(lime_image_path)
        plt.close(fig)

        explanation_list = explanation.as_list()
        max_impact_feature = max(explanation_list, key=lambda x: abs(x[1]))
        max_impact_text = "The feature that impacted the prediction the most is " + max_impact_feature[0] + " i.e, this above specified the feature within this specific range had the most significant impact on the model's prediction for the given instance."

    return render_template("index.html", prediction=output, lime_image=lime_image_path, max_impact_text=max_impact_text)

    

if __name__ == "__main__":
    app.run(debug = True)



# Flask
# Gunicorn
# pandas
# catboost
# shap
# matplotlib
# seaborn
# scikit-plot
# scikit-learn
# lime
# numpy