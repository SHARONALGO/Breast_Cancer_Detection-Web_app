import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import os
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import joblib
import subprocess
import logging
from werkzeug.utils import secure_filename
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key
from flask import flash, redirect, url_for, render_template, session, request, send_from_directory  # Add send_from_directory here
from werkzeug.utils import secure_filename
import os
import subprocess

UPLOAD_FOLDER = 'uploads'
IMAGE_UPLOAD_FOLDER = 'image_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['IMAGE_UPLOAD_FOLDER'] = IMAGE_UPLOAD_FOLDER

# Ensure the upload directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_UPLOAD_FOLDER, exist_ok=True)

# User login credentials
users = {
    'Dr Hanna Anilal': '12345',
    'Dr Divya': '54321'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username'].strip()
    password = request.form['password'].strip()

    if username in users and users[username] == password:
        session['username'] = username
        return redirect(url_for('options'))
    else:
        error = "Incorrect password. Please try again."
        return render_template('home.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))

@app.route('/options')
def options():
    if 'username' in session:
        return render_template('options.html', username=session['username'])
    else:
        return redirect(url_for('home'))

@app.route('/train', methods=['GET', 'POST'])
def train():
    if 'username' not in session:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Load the CSV file
            df = pd.read_csv(file_path)
            data_html = df.head(20).to_html(classes='table table-striped', index=False)
            
            # Render the data and show the Train button
            return render_template('train.html', data_html=data_html)
    return render_template('train.html')

@app.route('/train_model', methods=['POST'])
def train_model():
    if 'username' not in session:
        return redirect(url_for('home'))

    last_uploaded_file = sorted(os.listdir(app.config['UPLOAD_FOLDER']))[-1]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], last_uploaded_file)
    df = pd.read_csv(file_path)
    
    df = df.dropna(axis=1)
    y = df['diagnosis']
    x = df.drop(columns=["diagnosis", "id"], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    inst = DecisionTreeClassifier()
    parameters = {"max_depth": [1, 2, 3, 4, 5, 7, 10],
                  "min_samples_leaf": [1, 3, 6, 10, 20]}

    clf = GridSearchCV(inst, parameters, n_jobs=-1)
    clf.fit(x_train, y_train)
    global model
    model = clf.fit(x_train, y_train)
    prediction = model.predict(x_test)
    accuracy = accuracy_score(y_test, prediction)
    # Save the trained model
    joblib.dump(model, 'model.pkl')
    
    return jsonify({"status": "success", "accuracy": accuracy * 100, "best_params": clf.best_params_})

@app.route('/analysis')
def analysis():
    if 'username' not in session:
        return redirect(url_for('home'))
    
    # Load the data again for analysis
    last_uploaded_file = sorted(os.listdir(app.config['UPLOAD_FOLDER']))[-1]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], last_uploaded_file)
    df = pd.read_csv(file_path)
    
    # Calculate the percentages
    malignant_percentage = df[df['diagnosis'] == 'M'].shape[0] / df.shape[0] * 100
    benign_percentage = df[df['diagnosis'] == 'B'].shape[0] / df.shape[0] * 100
    
    return render_template('analysis.html', malignant_percentage=malignant_percentage, benign_percentage=benign_percentage)

@app.route('/about')
def about():
    if 'username' not in session:
        return redirect(url_for('home'))
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        try:
            # Collect input data from the form
            features = [
                float(request.form['radius_mean']),
                float(request.form['texture_mean']),
                float(request.form['perimeter_mean']),
                float(request.form['area_mean']),
                float(request.form['smoothness_mean']),
                float(request.form['compactness_mean']),
                float(request.form['concavity_mean']),
                float(request.form['concave_points_mean']),
                float(request.form['symmetry_mean']),
                float(request.form['fractal_dimension_mean']),
                float(request.form['radius_se']),
                float(request.form['texture_se']),
                float(request.form['perimeter_se']),
                float(request.form['area_se']),
                float(request.form['smoothness_se']),
                float(request.form['compactness_se']),
                float(request.form['concavity_se']),
                float(request.form['concave_points_se']),
                float(request.form['symmetry_se']),
                float(request.form['fractal_dimension_se']),
                float(request.form['radius_worst']),
                float(request.form['texture_worst']),
                float(request.form['perimeter_worst']),
                float(request.form['area_worst']),
                float(request.form['smoothness_worst']),
                float(request.form['compactness_worst']),
                float(request.form['concavity_worst']),
                float(request.form['concave_points_worst']),
                float(request.form['symmetry_worst']),
                float(request.form['fractal_dimension_worst'])
            ]
            
            # Use the model to predict
            prediction = model.predict([features])[0]
            result = 'Malignant' if prediction == 'M' else 'Benign'
            
            return render_template('predict_result.html', username=session['username'], result=result)
        
        except Exception as e:
            flash(f"An error occurred: {str(e)}")
            return redirect(url_for('options'))

    return render_template('predict.html', username=session['username'])

UPLOAD_FOLDER = 'C:\\Users\\user\\Desktop\\cancer_detection\\image_detection\\tf\\input'
OUTPUT_FOLDER = 'C:\\Users\\user\\Desktop\\cancer_detection\\image_detection\\tf\\output'

app.config['IMAGE_UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure input and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# Setup logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s: %(message)s')


from werkzeug.utils import secure_filename

@app.route('/detect_from_images', methods=['GET', 'POST'])
def detect_from_images():
    if 'username' not in session:
        return redirect(url_for('home'))

    if request.method == 'POST':
        file = request.files.get('file')
        
        if not file or file.filename == '':
            flash("No file uploaded or invalid file. Please try again.")
            return redirect(url_for('detect_from_images'))

        print(f"Received file: {file.filename}")

        if file.filename.endswith('.png'):
            # Sanitize the filename
            filename = secure_filename(file.filename)
            print(f"Sanitized filename: {filename}")

            # Define paths for input and output
            input_folder = app.config['IMAGE_UPLOAD_FOLDER']  # Use the folder instead of a file
            input_path = os.path.join(input_folder, filename)  # This is the full path for saving the input image
            output_path = os.path.join(app.config['OUTPUT_FOLDER'])

            try:
                print(f"Saving file to: {input_path}")
                file.save(input_path)
                print("File saved successfully.")
            except Exception as e:
                print(f"Error saving file: {e}")
                flash("Error saving the file.")
                return redirect(url_for('detect_from_images'))

            # Paths for the predict.py script and the virtual environment executable
            predict_script = r'C:\Users\user\Desktop\cancer_detection\image_detection\tf\predict.py'
            python_executable = r'C:\Users\user\Desktop\cancer_detection\hanna\Scripts\python.exe'
            model_path = r'C:\Users\user\Desktop\cancer_detection\image_detection\tf\saved_model'
            label_map_path = r'C:\Users\user\Desktop\cancer_detection\image_detection\tf\label_map.pbtxt'

            # Prepare the command to run
            command = [
                python_executable, predict_script,
                '--input', input_folder,  # Use the folder as input
                '--output', output_path,
                '--threshold', '0.7',
                '--model', model_path,
                '--label', label_map_path
            ]
            
            # Log the command for debugging
            print("Subprocess command:", ' '.join(command))

            # Run predict.py script with required arguments
            try:
                print("Running predict.py script...")
                result = subprocess.run(command, check=True, capture_output=True, text=True)

                # Log subprocess output and error for debugging
                print("Subprocess return code:", result.returncode)
                print("Subprocess stdout:", result.stdout)
                print("Subprocess stderr:", result.stderr)

                # Check if the output file was created
                if os.path.exists(output_path):
                    print(f"Output file found at: {output_path}")
                    return render_template('image_result.html', username=session['username'], output_image=filename)
                else:
                    print("Output file not found.")
                    flash("Prediction failed. Please try again.")
                    return redirect(url_for('detect_from_images'))

            except subprocess.CalledProcessError as e:
                print(f"Error running predict.py: {e.stderr}")
                flash(f"An error occurred while running the prediction: {e}")
                return redirect(url_for('detect_from_images'))
        else:
            flash("Please upload a valid PNG image.")
            return redirect(url_for('detect_from_images'))

    return render_template('detect_from_images.html', username=session['username'])




@app.route('/output_image/<filename>')
def output_image(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
