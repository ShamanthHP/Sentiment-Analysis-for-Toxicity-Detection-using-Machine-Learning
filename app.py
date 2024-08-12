
from flask import Flask, render_template, url_for, request, jsonify      
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

# Load the TF-IDF vocabulary specific to the category
with open(r"pickle/toxic_vect.pkl", "rb") as f:
    tox = pickle.load(f)

with open(r"pickle/severe_toxic_vect.pkl", "rb") as f:
    sev = pickle.load(f)

with open(r"pickle/obscene_vect.pkl", "rb") as f:
    obs = pickle.load(f)

with open(r"pickle/insult_vect.pkl", "rb") as f:
    ins = pickle.load(f)

with open(r"pickle/threat_vect.pkl", "rb") as f:
    thr = pickle.load(f)

with open(r"pickle/identity_hate_vect.pkl", "rb") as f:
    ide = pickle.load(f)

# Load the pickled RDF models
with open(r"pickle/toxic_model.pkl", "rb") as f:
    tox_model = pickle.load(f)

with open(r"pickle/severe_toxic_model.pkl", "rb") as f:
    sev_model = pickle.load(f)

with open(r"pickle/obscene_model.pkl", "rb") as f:
    obs_model  = pickle.load(f)

with open(r"pickle/insult_model.pkl", "rb") as f:
    ins_model  = pickle.load(f)

with open(r"pickle/threat_model.pkl", "rb") as f:
    thr_model  = pickle.load(f)

with open(r"pickle/identity_hate_model.pkl", "rb") as f:
    ide_model  = pickle.load(f)



@app.route("/")
@app.route("/index")
def index():
	return render_template('index.html')
@app.route('/login')
def login():
	return render_template('login.html')    
    

@app.route('/chart')
def chart():
	return render_template('chart.html')

@app.route('/performance')
def performance():
	return render_template('performance.html')    


@app.route('/upload')
def upload():
    return render_template('upload.html')  
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)	

@app.route("/toxic")
def toxic():
    return render_template('toxic.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    # Take a string input from user
    user_input = request.form['text']
    data = [user_input]

    vect = tox.transform(data)
    pred_tox = tox_model.predict_proba(vect)[:,1]

    vect = sev.transform(data)
    pred_sev = sev_model.predict_proba(vect)[:,1]

    vect = obs.transform(data)
    pred_obs = obs_model.predict_proba(vect)[:,1]

    vect = thr.transform(data)
    pred_thr = thr_model.predict_proba(vect)[:,1]

    vect = ins.transform(data)
    pred_ins = ins_model.predict_proba(vect)[:,1]

    vect = ide.transform(data)
    pred_ide = ide_model.predict_proba(vect)[:,1]

    out_tox = round(pred_tox[0], 2)
    out_sev = round(pred_sev[0], 2)
    out_obs = round(pred_obs[0], 2)
    out_ins = round(pred_ins[0], 2)
    out_thr = round(pred_thr[0], 2)
    out_ide = round(pred_ide[0], 2)

    print(out_tox)

    return render_template('toxic.html', 
                            pred_tox = 'Toxic: {}'.format(out_tox),
                            pred_sev = 'Severe Toxic: {}'.format(out_sev), 
                            pred_obs = 'Obscene: {}'.format(out_obs),
                            pred_ins = 'Insult: {}'.format(out_ins),
                            pred_thr = 'Threat: {}'.format(out_thr),
                            pred_ide = 'Identity Hate: {}'.format(out_ide)                        
                            )
     
# Server reloads itself if code changes so no need to keep restarting:
if __name__ == "__main__":
    app.run(debug=True)

