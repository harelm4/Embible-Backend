from flask import Flask, request,jsonify
from flask_cors import CORS
import config
from src.model.ensemble import Ensemble
from huggingface_hub import login


app = Flask(__name__)
CORS(app)
login(config.configs['hf_token'])
print('initializing ensemble models')
ens=Ensemble()
print('starting flask')
@app.route("/calc",methods=['GET'])
def calc():
     args=request.args.to_dict()
     return jsonify(ens.predict(args['text']).get_ui_format())

app.run(host="0.0.0.0", port=443,debug=True,ssl_context='adhoc') #server
# app.run(host="0.0.0.0", port=8000,debug=True) #local
