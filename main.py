from flask import Flask, render_template, request,jsonify
import flask
from flask_cors import CORS, cross_origin
import config
from huggingface_hub import login

from src.model.ensemble_v2 import EnsembleV2

app = Flask(__name__)
CORS(app)
login(config.configs['hf_token'])
ens=EnsembleV2()

@app.route("/calc",methods=['GET'])
def calc():
    args=request.args.to_dict()
    response = flask.jsonify(ens.predict(args['text']))
    return response

@app.route("/")
def hello():
    return  render_template("index.html")

app.run(host="0.0.0.0", port=443,debug=True,ssl_context='adhoc')


