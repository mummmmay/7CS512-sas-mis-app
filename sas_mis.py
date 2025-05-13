from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify, send_file
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

        # Clear previously conditioned data if any
        conditioned_path = os.path.join(UPLOAD_FOLDER, 'conditioned_data.pkl')
        if os.path.exists(conditioned_path):
            os.remove(conditioned_path)

        flash("✅ Columns saved and conditioned data cleared. Continue to the next step.", "success")
        return redirect('/conditions')

    return render_template('columns.html', columns=columns_info)
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
def conditional_builder():
    try:
        df = pd.read_pickle(os.path.join(UPLOAD_FOLDER, 'selected_data.pkl'))
    except Exception:
        flash("⚠️ Dataset not found. Please complete upload and column selection first.", "danger")
        return redirect('/upload')

    preview_table = None
    all_inputs = request.form.to_dict(flat=False)
    added_any_column = False

    if request.method == 'POST':
        group_indices = all_inputs.get("group_index[]", [])
        for idx, group_id in enumerate(group_indices):
            col_name = all_inputs.get(f"column_name_{group_id}", [""])[0].strip()
            val_true = all_inputs.get(f"value_true_{group_id}", [""])[0].strip()
            val_false = all_inputs.get(f"value_false_{group_id}", [""])[0].strip()
            fields = all_inputs.get(f"field_{group_id}[]", [])
            operators = all_inputs.get(f"operator_{group_id}[]", [])
            values = all_inputs.get(f"value_{group_id}[]", [])
            logics = all_inputs.get(f"logic_{group_id}[]", [])

            if not col_name or not val_true or not val_false:
                flash(f"⚠️ Skipping column {idx+1}: missing name or true/false values.", "warning")
                continue

            if not fields or not operators or not values:
                flash(f"⚠️ Skipping column '{col_name}': missing condition rules.", "warning")
                continue

            conditions = []
            for j in range(len(fields)):
                field = fields[j].strip()
                op = operators[j].strip()
                val = values[j].strip()
                logic = logics[j].strip().lower() if j < len(logics) else ""

                if not field or not op or not val:
                    continue

                try:
                    val = float(val) if val.replace('.', '', 1).isdigit() else val
                except:
                    pass

                if op == "contains":
                    cond = f"(df['{field}'].astype(str).str.contains(r'{val}'))"
                elif op == "startswith":
                    cond = f"(df['{field}'].astype(str).str.startswith('{val}'))"
                elif op == "endswith":
                    cond = f"(df['{field}'].astype(str).str.endswith('{val}'))"
                else:
                    cond = f"(df['{field}'] {op} {repr(val)})"

                if j > 0 and logic in ['and', 'or']:
                    cond = f"{logic} {cond}"
                conditions.append(cond)

            if not conditions:
                flash(f"⚠️ No valid rules defined for '{col_name}'.", "warning")
                continue

            full_expr = " ".join(conditions)
            try:
                result = eval(full_expr)
                df[col_name] = result.map({True: val_true, False: val_false})
                added_any_column = True
            except Exception as e:
                flash(f"❌ Failed to evaluate logic for '{col_name}': {e}", "danger")

        if added_any_column:
            df.to_pickle(os.path.join(UPLOAD_FOLDER, 'conditioned_data.pkl'))
            preview_table = df.head(10).to_html(classes="table table-bordered table-hover", index=False)
            flash("✅ Column(s) successfully added!", "success")
        else:
            flash("⚠️ No columns were added. Please check your inputs.", "warning")

    column_types = dict(df.dtypes.apply(lambda d: d.name))
    unique_values = {col: df[col].dropna().astype(str).unique().tolist() for col in df.columns}

    return render_template(
        'conditions.html',
        columns=list(df.columns),
        column_types=column_types,
        unique_values=unique_values,
        preview_table=preview_table
    )

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

    # Save as CSV and also as PKL for /config use
    df_sample.to_csv(filepath, index=False)
    df_sample.to_pickle(os.path.join(UPLOAD_FOLDER, 'sampled_data.pkl'))

    summary_html = df_sample.describe(include='all').transpose().head(10).to_html(classes='table table-bordered')
    preview_html = df_sample.head(10).to_html(
        classes="table table-striped table-bordered table-sm text-center align-middle", 
        index=False,
        border=0
    )

    return render_template('sample.html', n=n, summary=summary_html, preview=preview_html, filename=filename)

@app.route('/config', methods=['GET', 'POST'])
def config_page():
    try:
        df = pd.read_pickle(os.path.join(UPLOAD_FOLDER, 'sampled_data.pkl'))
    except Exception:
        flash("⚠️ Please sample a dataset before configuring SAS procedures.", "warning")
        return redirect('/sample')

    if request.method == 'POST':
        selected_procs = []
        for key, value in request.form.items():
            if key.startswith('variable_'):
                proc_name = key.replace('variable_', '')
                variables = request.form.getlist(key)
                if variables:
                    selected_procs.append({'procedure': proc_name, 'variables': variables})
            elif key == 'custom_proc':
                custom_proc = value.strip()
                custom_vars = request.form.getlist('custom_variables')
                if custom_proc and custom_vars:
                    selected_procs.append({'procedure': custom_proc, 'variables': custom_vars})

        if selected_procs:
            import csv
            config_path = os.path.join(UPLOAD_FOLDER, 'config.csv')
            with open(config_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['procedure', 'variables'])
                for entry in selected_procs:
                    writer.writerow([entry['procedure'], ','.join(entry['variables'])])
            flash("✅ Config.csv generated successfully!", "success")
            return send_file(config_path, as_attachment=True)
        else:
            flash("⚠️ No procedures or variables selected.", "warning")

    default_procs = {
        "BASIC": {
            "PROC CONTENTS": "View metadata of dataset.",
            "PROC PRINT": "Print dataset rows.",
            "PROC MEANS": "Summary stats for numeric columns.",
            "PROC FREQ": "Frequency tables for categorical vars.",
            "PROC UNIVARIATE": "Detailed distribution analysis."
        },
        "ADVANCED": {
            "PROC LOGISTIC": "Binary classification.",
            "PROC REG": "Linear regression.",
            "PROC CLUSTER": "Customer segmentation.",
            "PROC FACTOR": "Factor analysis."
        },
        "VISUAL": {
            "PROC SGPLOT": "Bar, line, histogram, scatter.",
            "PROC SGSCATTER": "Scatter plots or matrices.",
            "PROC SGPANEL": "Facet/group plots."
        },
        "DATA_SQL": {
            "PROC SQL": "Run SQL-like queries.",
            "PROC SORT": "Sort dataset by variables.",
            "PROC TRANSPOSE": "Reshape dataset (wide/long)."
        }
    }

    return render_template('config.html', default_procs=default_procs, columns=list(df.columns))

@app.route('/download_config', methods=['POST'])
def download_config():
    preview_json = request.form.get('preview_rows')
    if not preview_json:
        flash("No config data to download.", "danger")
        return redirect('/config')

    try:
        preview_data = pd.DataFrame(json.loads(preview_json))
        preview_data = (
            preview_data.apply(lambda col: col.str.strip() if col.dtype == 'object' else col)
            .assign(analysis=lambda df: df['analysis'].str.upper())
            .groupby('analysis')['variable']
            .apply(lambda x: ' '.join(sorted(x.unique())))
            .reset_index()
        )
        config_path = os.path.join(DOWNLOAD_FOLDER, 'config.csv')
        preview_data.to_csv(config_path, index=False, encoding='utf-8')
        return send_from_directory(DOWNLOAD_FOLDER, 'config.csv', as_attachment=True)
    except Exception as e:
        flash(f"Error generating config file: {e}", "danger")
        return redirect('/config')

@app.route('/walkthrough')
def walkthrough():
    return render_template('walkthrough.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(DOWNLOAD_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)