from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify
import pandas as pd
import numpy as np
import os
import json

app = Flask(__name__)
app.secret_key = 'my_super_secret_key'

UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

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
            df = pd.read_csv(file, nrows=1000)
        elif filename.endswith('.xlsx'):
            df = pd.read_excel(file, nrows=1000)
        elif filename.endswith('.sas7bdat'):
            df = pd.read_sas(file)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        # Save full file
        file.stream.seek(0)
        if filename.endswith('.csv'):
            full_df = pd.read_csv(file)
        elif filename.endswith('.xlsx'):
            full_df = pd.read_excel(file)
        elif filename.endswith('.sas7bdat'):
            full_df = pd.read_sas(file)

        full_df.to_pickle(os.path.join(UPLOAD_FOLDER, 'full_data.pkl'))

        preview_html = df.head(10).to_html(classes='table table-bordered table-sm', index=False)
        return jsonify({'preview_html': preview_html})
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/columns', methods=['GET', 'POST'])
def column_selector():
    try:
        df = pd.read_pickle(os.path.join(UPLOAD_FOLDER, 'full_data.pkl'))
    except FileNotFoundError:
        flash("No dataset found. Please upload a file first.", "danger")
        return redirect('/upload')

    columns_info = [{"name": col, "dtype": str(dtype)} for col, dtype in df.dtypes.items()]
    if request.method == 'POST':
        selected_columns = request.form.getlist('columns')
        if not selected_columns:
            flash("Please select at least one column.", "warning")
            return redirect(request.url)

        df_filtered = df[selected_columns]
        df_filtered.to_pickle(os.path.join(UPLOAD_FOLDER, 'selected_data.pkl'))
        return redirect('/conditions')

    return render_template('columns.html', columns=columns_info)

@app.route('/conditions', methods=['GET', 'POST'])
def condition_builder():
    try:
        df = pd.read_pickle(os.path.join(UPLOAD_FOLDER, 'selected_data.pkl'))
    except FileNotFoundError:
        flash("No column-filtered dataset found. Please complete column selection first.", "danger")
        return redirect('/columns')

    column_names = df.columns.tolist()
    column_types = {col: "string" if df[col].dtype == 'object' else "numeric" for col in df.columns}
    unique_values = {col: df[col].dropna().astype(str).value_counts().head(5).index.tolist() for col in df.columns}
    preview_html = None

    if request.method == 'POST':
        conditions = request.form.getlist('condition')
        operators = request.form.getlist('operator')
        values = request.form.getlist('value')
        logics = request.form.getlist('logic')

        true_value = request.form.get('true_value')
        false_value = request.form.get('false_value')
        new_col_name = request.form.get('new_col_name')

        if not new_col_name:
            flash("Please enter a name for the new column.", "warning")
            return redirect(request.url)

        expression_parts = []
        for i in range(len(conditions)):
            col = conditions[i].strip()
            op = operators[i].strip()
            val = values[i].strip()
            logic = logics[i].strip().lower() if i < len(logics) else ''

            if not col or not op or not val:
                continue

            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                try:
                    val = float(val)
                    part = f"(df['{col}'] {op} {val})"
                except ValueError:
                    flash(f"'{val}' is not a valid number for column '{col}'.", "danger")
                    return redirect(request.url)
            else:
                val = str(val)
                if op == 'contains':
                    part = f"(df['{col}'].str.contains({repr(val)}, na=False))"
                elif op == 'startswith':
                    part = f"(df['{col}'].str.startswith({repr(val)}, na=False))"
                elif op == 'endswith':
                    part = f"(df['{col}'].str.endswith({repr(val)}, na=False))"
                elif op in ['==', '!=']:
                    part = f"(df['{col}'] {op} {repr(val)})"
                else:
                    flash(f"Unsupported operator '{op}' for column '{col}'.", "danger")
                    return redirect(request.url)

            expression_parts.append(part)
            if logic == "and":
                expression_parts.append("&")
            elif logic == "or":
                expression_parts.append("|")

        if not expression_parts:
            flash("Please enter at least one valid condition.", "warning")
            return redirect(request.url)

        full_condition = ' '.join(expression_parts)
        try:
            condition_result = eval(f"({full_condition})")
            df[new_col_name] = np.where(condition_result, true_value, false_value)
            df.to_pickle(os.path.join(UPLOAD_FOLDER, 'conditioned_data.pkl'))
            preview_html = df[[new_col_name]].head(5).to_html(classes="table table-bordered", index=False)
        except Exception as e:
            flash(f"Error in condition logic: {e}", "danger")
            return redirect(request.url)

    return render_template('conditions.html', columns=column_names, preview=preview_html, unique_values=unique_values, column_types=column_types)

@app.route('/sample')
def sample_dataset():
    try:
        df = pd.read_pickle(os.path.join(UPLOAD_FOLDER, 'conditioned_data.pkl'))
    except FileNotFoundError:
        flash("Conditional dataset not found. Please complete previous steps first.", "danger")
        return redirect('/conditions')

    n = 20000 if len(df) >= 20000 else len(df)
    df_sample = df.sample(n=n, random_state=42)
    filename = 'dataset_sampling.csv'
    filepath = os.path.join(DOWNLOAD_FOLDER, filename)
    df_sample.to_csv(filepath, index=False)

    summary_html = df_sample.describe(include='all').transpose().head(10).to_html(classes='table table-bordered')
    preview_html = df_sample.head(10).to_html(classes='table table-striped table-sm', index=False)

    return render_template('sample.html', n=n, summary=summary_html, preview=preview_html, filename=filename)

@app.route('/config', methods=['GET', 'POST'])
def config_generator():
    try:
        df = pd.read_pickle(os.path.join(UPLOAD_FOLDER, 'conditioned_data.pkl'))
    except FileNotFoundError:
        flash("Processed dataset not found. Please complete the previous steps.", "danger")
        return redirect('/conditions')

    columns = df.columns.tolist()
    default_procs = ['proc_freq', 'proc_means', 'proc_logistic']
    preview_data = None

    if request.method == 'POST':
        procedures = request.form.getlist('procedure')
        custom_procs = request.form.getlist('custom_proc')
        rows = []

        for i in range(len(procedures)):
            proc = procedures[i].strip()
            if proc == 'custom':
                proc = custom_procs[i].strip().lower()
            else:
                proc = proc.lower()

            selected_vars = request.form.getlist(f'variable_{i}')
            for var in selected_vars:
                rows.append({'analysis': proc, 'variable': var})

        if rows:
            preview_data = pd.DataFrame(rows)
        else:
            flash("Please select at least one variable.", "warning")

    return render_template('config.html', columns=columns, default_procs=default_procs, preview_data=preview_data.to_dict(orient='records') if preview_data is not None else None)

@app.route('/download_config', methods=['POST'])
def download_config():
    preview_json = request.form.get('preview_rows')
    if not preview_json:
        flash("No config data to download.", "danger")
        return redirect('/config')

    preview_data = pd.DataFrame(json.loads(preview_json))
    config_path = os.path.join(DOWNLOAD_FOLDER, 'config.csv')
    preview_data.to_csv(config_path, index=False)

    return send_from_directory(DOWNLOAD_FOLDER, 'config.csv', as_attachment=True)

@app.route('/walkthrough')
def walkthrough():
    return render_template('walkthrough.html')

if __name__ == "__main__":
    app.run(debug=True)
