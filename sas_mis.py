from flask import Flask, render_template, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload_ajax', methods=['POST'])
def upload_ajax():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    filename = file.filename.lower()

    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file)
        elif filename.endswith('.sas7bdat'):
            df = pd.read_sas(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        preview_html = df.head(10).to_html(classes='table table-bordered table-sm', index=False)
        df.to_pickle(os.path.join(UPLOAD_FOLDER, 'full_data.pkl'))

        return jsonify({'preview_html': preview_html})

    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/walkthrough')
def walkthrough():
    return render_template('walkthrough.html')

if __name__ == "__main__":
    app.run(debug=True)
