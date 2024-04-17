from flask import Flask, request, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate():
    file = request.files['dataset']
    file.save('train.csv')  # Save the uploaded dataset file

    # Execute the Python script and capture the output
    output = subprocess.check_output(['python', 'Demand Forcasting for a Retail Store(1).py'])

    return output

if __name__ == '__main__':
    app.run(debug=True)
