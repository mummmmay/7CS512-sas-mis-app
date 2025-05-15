from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify, send_file
from zipfile import ZipFile
from datetime import datetime
from collections import defaultdict
import io
import pandas as pd
import numpy as np
import os
import json
import csv


app = Flask(__name__)
app.secret_key = 'my_super_secret_key'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOWNLOAD_FOLDER'] = 'downloads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)
# os.environ["EMAIL_RECIPIENT"] = form.get('email_recipient')

@app.context_processor
def inject_now():
    return {'now': datetime.now}

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

        df.to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.pkl'))
        preview_html = df.head(10).to_html(classes="table table-bordered table-striped", index=False)
        return jsonify({'preview_html': preview_html})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/columns', methods=['GET', 'POST'])
def columns():
    data_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_data.pkl')
    if not os.path.exists(data_path):
        flash("⚠️ Please upload a dataset first.", "warning")
        return redirect('/upload')

    df = pd.read_pickle(data_path)

    if request.method == 'POST':
        selected_columns = request.form.getlist('columns')
        df_filtered = df[selected_columns]
        df_filtered.to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_data.pkl'))
        return redirect('/conditions')

    columns = [{'name': col, 'dtype': str(df[col].dtype)} for col in df.columns]
    return render_template('columns.html', columns=columns)

@app.route('/conditions', methods=['GET', 'POST'])
def conditions():
    df = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_data.pkl'))
    columns = df.columns.tolist()
    preview_table = None

    if request.method == 'POST':
        group_indexes = request.form.getlist('group_index[]')

        if not group_indexes:
            df.to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'conditioned_data.pkl'))
            flash("✅ No conditions added. Proceeding with selected columns only.", "info")
            return redirect('/sample')

        for group_index in group_indexes:
            col_name = request.form.get(f'column_name_{group_index}')
            value_true = request.form.get(f'value_true_{group_index}')
            value_false = request.form.get(f'value_false_{group_index}')
            fields = request.form.getlist(f'field_{group_index}[]')
            operators = request.form.getlist(f'operator_{group_index}[]')
            values = request.form.getlist(f'value_{group_index}[]')
            logics = request.form.getlist(f'logic_{group_index}[]')

            condition_str = ''
            for i in range(len(fields)):
                field = fields[i]
                operator_ = operators[i]
                value = values[i]
                logic = logics[i - 1] if i > 0 and i - 1 < len(logics) else ''

                col_dtype = df[field].dtype
                try:
                    if col_dtype in ['int64', 'float64']:
                        value_casted = float(value)
                    elif col_dtype == 'bool':
                        value_casted = value.lower() in ['true', '1', 'yes']
                    else:
                        value_casted = f'"{value}"'
                except:
                    value_casted = f'"{value}"'

                part = f"(df['{field}'] {operator_} {value_casted})"
                condition_str += part if i == 0 else f" {logic} {part}"

            try:
                df[col_name] = np.where(eval(condition_str), value_true, value_false)
            except Exception as e:
                flash(f"⚠️ Error in condition for '{col_name}': {e}", "danger")

        df.to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'conditioned_data.pkl'))
        flash("✅ Conditioned data saved successfully.", "success")
        preview_table = df.head(10).to_html(classes='table table-bordered table-sm', index=False)

    return render_template('conditions.html', columns=columns, preview_table=preview_table)

@app.route('/skip-conditions')
def skip_conditions():
    path_filtered = os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_data.pkl')
    path_conditioned = os.path.join(app.config['UPLOAD_FOLDER'], 'conditioned_data.pkl')

    try:
        if os.path.exists(path_filtered):
            df = pd.read_pickle(path_filtered)
            df.to_pickle(path_conditioned)
            flash("⏭️ Skipped conditions — using selected columns only.", "info")
        else:
            flash("⚠️ No filtered data found. Please select columns first.", "warning")
            return redirect('/columns')
    except Exception as e:
        flash(f"❌ Error skipping: {e}", "danger")
        return redirect('/columns')

    return redirect('/sample')

@app.route('/sample', methods=['GET', 'POST'])
def sample_dataset():
    summary_html = None
    preview_html = None
    df_sample = None

    path_filtered = os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_data.pkl')
    path_conditioned = os.path.join(app.config['UPLOAD_FOLDER'], 'conditioned_data.pkl')
    sample_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sampled_data.pkl')
    csv_path = os.path.join(app.config['DOWNLOAD_FOLDER'], 'dataset_sampling.csv')

    try:
        if os.path.exists(path_conditioned):
            df = pd.read_pickle(path_conditioned)
            source = "conditioned_data.pkl"
        elif os.path.exists(path_filtered):
            df = pd.read_pickle(path_filtered)
            source = "filtered_data.pkl"
        else:
            flash("🛑 Missing both filtered and conditioned data. Please select columns first.", "danger")
            return redirect('/columns')

        df_sample = df.sample(n=min(20000, len(df)), random_state=42)
        df_sample.to_pickle(sample_path)
        df_sample.to_csv(csv_path, index=False)

        pd.set_option("display.max_columns", None)
        summary_html = df_sample.describe(include='all').transpose().head(10).to_html(classes='table table-bordered')
        preview_html = df_sample.head(10).to_html(classes="table table-striped table-bordered", index=False)

        flash(f"✅ Sampled {len(df_sample)} rows from {source}", "success")

    except Exception as e:
        flash(f"❌ Error: {e}", "danger")
        return redirect('/columns')

    return render_template('sample.html', summary=summary_html, preview=preview_html, filename='dataset_sampling.csv', n=len(df_sample))

@app.route('/downloads/<filename>')
def download_file(filename):
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], filename, as_attachment=True)

@app.route('/config-analysis', methods=['GET', 'POST'])
def config_analysis():
    sampled_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sampled_data.pkl')
    if not os.path.exists(sampled_path):
        flash("⚠️ Please sample the dataset first.", "warning")
        return redirect('/sample')

    df = pd.read_pickle(sampled_path)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    binary_columns = [col for col in df.columns if df[col].nunique() == 2]

    if request.method == 'POST':
        config_map = defaultdict(list)

        def add(proc, value):
            if value:
                config_map[f"PROC {proc}"].append(value)

        # Collect input fields
        for col in request.form.getlist('means_columns'):
            add('MEANS', col)

        for col in request.form.getlist('freq_columns'):
            add('FREQ', col)

        add('LOGISTIC', request.form.get('logistic_target'))
        for col in request.form.getlist('logistic_predictors'):
            add('LOGISTIC', col)

        sql = request.form.get('sql_query')
        if sql:
            add('SQL', sql.strip())

        for col in request.form.getlist('corr_columns'):
            add('CORR', col)

        add('REG', request.form.get('reg_target'))
        for col in request.form.getlist('reg_predictors'):
            add('REG', col)

        for col in request.form.getlist('univariate_columns'):
            add('UNIVARIATE', col)

        x, y = request.form.get('sgplot_x'), request.form.get('sgplot_y')
        if x and y:
            add('SGPLOT', x)
            add('SGPLOT', y)

        add('GLM', request.form.get('glm_target'))
        for col in request.form.getlist('glm_predictors'):
            add('GLM', col)

        t, y = request.form.get('arima_time'), request.form.get('arima_target')
        if t and y:
            add('ARIMA', t)
            add('ARIMA', y)

        for col in request.form.getlist('cluster_columns'):
            add('CLUSTER', col)

        add('HPFOREST', request.form.get('hpforest_target'))
        for col in request.form.getlist('hpforest_predictors'):
            add('HPFOREST', col)

        if not config_map:
            flash("⚠️ No PROC configuration selected.", "warning")
            return redirect('/config-analysis')

        # Save to JSON
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'user_config.json'), 'w') as f:
            json.dump(dict(config_map), f)

        # Save to CSV
        with open(os.path.join(app.config['DOWNLOAD_FOLDER'], 'config_analysis.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['procedure', 'variables'])
            for proc, values in config_map.items():
                writer.writerow([proc, ' '.join(values)])

        # Create HTML preview
        preview_html = pd.DataFrame([
            {'procedure': proc, 'variables': ' '.join(values)}
            for proc, values in config_map.items()
        ]).to_html(classes="table table-bordered table-sm", index=False)

        flash("✅ All PROC configurations saved successfully!", "success")
        return render_template(
            'config-analysis.html',
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            binary_columns=binary_columns,
            df=df,
            preview_html=preview_html,
            download_link="/download/config_analysis.csv"
        )

    # GET request fallback
    return render_template(
        'config-analysis.html',
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        binary_columns=binary_columns,
        df=df,
        preview_html=None,
        download_link=None
    )

@app.route('/save-config', methods=['POST'])
def save_config():
    try:
        config_path = os.path.join(app.config['UPLOAD_FOLDER'], 'user_config.json')
        if not os.path.exists(config_path):
            return "No saved config", 400

        with open(config_path, "w", newline="", encoding="utf-8") as f:
            config_map = json.load(f)

        with open(os.path.join(app.config['DOWNLOAD_FOLDER'], 'config_analysis.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['procedure', 'variables'])
            for proc, values in config_map.items():
                writer.writerow([proc, ' '.join(values)])

        return "Saved", 200
    except Exception as e:
        return f"Error: {e}", 500

@app.route('/save-visual-config', methods=['POST'])
def save_visual_config():
    sampled_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sampled_data.pkl')
    if not os.path.exists(sampled_path):
        return "Sampled data not found", 400

    df = pd.read_pickle(sampled_path)
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    config_path = os.path.join(app.config['UPLOAD_FOLDER'], 'user_visual_config.json')
    if not os.path.exists(config_path):
        return "No saved visual config", 400

    with open(config_path) as f:
        visual_map = json.load(f)

    try:

        # Save visual_map to JSON for later download use
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'user_visual_config.json'), 'w') as f_json:
            json.dump({k: v for k, v in visual_map.items()}, f_json)

        with open(os.path.join(app.config['DOWNLOAD_FOLDER'], 'config_visual.csv'), 'w', newline='') as f:
        
            writer = csv.writer(f)
            writer.writerow(['procedure', 'variables'])
            for proc, vars in visual_map.items():
                writer.writerow([proc, ' '.join(vars)])
        return "Saved", 200
    except Exception as e:
        return f"Error: {e}", 500

@app.route("/config-visual", methods=["GET", "POST"])
def config_visual():
    sampled_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sampled_data.pkl')
    if not os.path.exists(sampled_path):
        flash("⚠️ Please sample the dataset first.", "warning")
        return redirect("/sample")

    df = pd.read_pickle(sampled_path)
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if request.method == "POST":
        config = []

        def add_rows(proc, key):
            for value in request.form.getlist(key):
                config.append({"PROC": proc, "Variable": value})

        add_rows("MEANS", "means_column")
        add_rows("FREQ", "freq_column")
        add_rows("UNIVARIATE", "univariate_column")
        add_rows("CLUSTER", "cluster_column")

        if request.form.get("corr_x") and request.form.get("corr_y"):
            config.append({"PROC": "CORR", "Variable": f"{request.form['corr_x']} vs {request.form['corr_y']}"})

        if request.form.get("sgplot_x") and request.form.get("sgplot_y"):
            config.append({"PROC": "SGPLOT", "Variable": f"{request.form['sgplot_x']} vs {request.form['sgplot_y']}"})

        if request.form.get("glm_effect"):
            config.append({"PROC": "GLM", "Variable": request.form["glm_effect"]})

        if request.form.get("arima_column"):
            config.append({"PROC": "ARIMA", "Variable": request.form["arima_column"]})

        if request.form.get("hpforest_option"):
            config.append({"PROC": "HPFOREST", "Variable": request.form["hpforest_option"]})

        if request.form.get("logistic_target") and request.form.get("logistic_predictor"):
            config.append({
                "PROC": "LOGISTIC",
                "Variable": f"{request.form['logistic_predictor']} → {request.form['logistic_target']}"
            })

        config_path = os.path.join(app.config['DOWNLOAD_FOLDER'], 'config_visual.csv')
        with open(config_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["PROC", "Variable"])
            writer.writeheader()
            writer.writerows(config)

        flash("✅ Visual configuration saved successfully.", "success")
        return redirect("/config-visual")

    preview_data = df.head(200).to_json(orient="records")

    return render_template(
        "config-visual.html",
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        preview_data=preview_data
    )


@app.route('/walkthrough')
def walkthrough():
    return render_template('walkthrough.html')

from zipfile import ZipFile

@app.route('/download-all')
def download_all_files():
    zip_filename = "sas_output_bundle.zip"
    zip_path = os.path.join(app.config['DOWNLOAD_FOLDER'], zip_filename)

    # Clean old ZIP file if exists
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # Paths to target files
    files_to_include = {
        "sampled_data.csv": os.path.join(app.config['UPLOAD_FOLDER'], "dataset_sampling.csv"),
        "config_analysis.csv": os.path.join(app.config['DOWNLOAD_FOLDER'], "config_analysis.csv"),
        "config_visual.csv": os.path.join(app.config['DOWNLOAD_FOLDER'], "config_visual.csv"),
    }

    # Create new ZIP
    with ZipFile(zip_path, 'w') as zipf:
        for filename, fullpath in files_to_include.items():
            if os.path.exists(fullpath):
                zipf.write(fullpath, arcname=filename)

    return send_from_directory(app.config['DOWNLOAD_FOLDER'], zip_filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
