'''
Created on Apr 4, 2018

@author: animesh
'''
import os
from flask import Flask, jsonify, request
from flask_cors import CORS

PATH = os.getcwd()
app = Flask(__name__)    
CORS(app)


@app.route("/login", methods=["GET", "POST"])
def login():
    data = request.form.to_dict()
    print(data)
    result = "Login"
    return result
    
if __name__ == "__main__":
    app.run()