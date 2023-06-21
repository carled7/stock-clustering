from flask import Flask, render_template, request
from service import main

app = Flask(__name__, static_folder='static')


@app.route('/')
def index():
    print('running')
    return render_template('index.html', data='Hello, world')

@app.route('/process_data', methods=['POST'])
def process_data():
    input_data = request.form['input_data']
    pred_cluster = main(ticker_name=input_data)
    return render_template('result.html', ticker = input_data, cluster = pred_cluster)

