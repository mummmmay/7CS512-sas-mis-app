from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory, jsonify, send_file
import pandas as pd
import numpy as np
import os
import json
import csv
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'my_super_secret_key'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOWNLOAD_FOLDER'] = 'downloads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

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
    # Load previously filtered dataset
    df = pd.read_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_data.pkl'))
    columns = df.columns.tolist()
    preview_table = None

    if request.method == 'POST':
        group_indexes = request.form.getlist('group_index[]')

        if not group_indexes:
            # No new conditional columns added — just copy filtered to conditioned
            df.to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'conditioned_data.pkl'))
            flash("✅ No conditions added. Proceeding with selected columns only.", "info")
            return redirect('/sample')

        # Add derived columns per group
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

            # Apply condition
            try:
                df[col_name] = np.where(eval(condition_str), value_true, value_false)
            except Exception as e:
                flash(f"⚠️ Error in condition for '{col_name}': {e}", "danger")

        # Save updated dataset
        df.to_pickle(os.path.join(app.config['UPLOAD_FOLDER'], 'conditioned_data.pkl'))
        flash("✅ Conditioned data saved successfully.", "success")

        # Optional preview
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

        # Sample
        df_sample = df.sample(n=min(20000, len(df)), random_state=42)
        df_sample.to_pickle(sample_path)
        df_sample.to_csv(csv_path, index=False)

        # Preview + summary
        pd.set_option("display.max_columns", None)
        summary_html = df_sample.describe(include='all').transpose().head(10).to_html(classes='table table-bordered')
        preview_html = df_sample.head(10).to_html(classes="table table-striped table-bordered", index=False)

        flash(f"✅ Sampled {len(df_sample)} rows from {source}", "success")

    except Exception as e:
        flash(f"❌ Error: {e}", "danger")
        return redirect('/columns')

    return render_template('sample.html', summary=summary_html, preview=preview_html, filename='dataset_sampling.csv', n=len(df_sample))

@app.route('/download/<filename>')
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
        config = []

        # PROC MEANS
        for col in request.form.getlist('means_columns'):
            config.append({'PROC': 'MEANS', 'Column': col})

        # PROC FREQ
        for col in request.form.getlist('freq_columns'):
            config.append({'PROC': 'FREQ', 'Column': col})

        # PROC LOGISTIC
        target = request.form.get('logistic_target')
        predictors = request.form.getlist('logistic_predictors')
        if target:
            config.append({'PROC': 'LOGISTIC', 'Column': f"Target: {target}"})
            for col in predictors:
                config.append({'PROC': 'LOGISTIC', 'Column': col})

        # PROC SQL
        sql = request.form.get('sql_query')
        if sql:
            config.append({'PROC': 'SQL', 'Column': sql.strip()})

        # PROC CORR
        for col in request.form.getlist('corr_columns'):
            config.append({'PROC': 'CORR', 'Column': col})

        # PROC REG
        reg_target = request.form.get('reg_target')
        reg_predictors = request.form.getlist('reg_predictors')
        if reg_target:
            config.append({'PROC': 'REG', 'Column': f"Target: {reg_target}"})
            for col in reg_predictors:
                config.append({'PROC': 'REG', 'Column': col})

        # PROC UNIVARIATE
        for col in request.form.getlist('univariate_columns'):
            config.append({'PROC': 'UNIVARIATE', 'Column': col})

        # PROC SGPLOT
        sgplot_x = request.form.get('sgplot_x')
        sgplot_y = request.form.get('sgplot_y')
        if sgplot_x and sgplot_y:
            config.append({'PROC': 'SGPLOT', 'Column': f"X: {sgplot_x}, Y: {sgplot_y}"})

        # Save config file
        config_df = pd.DataFrame(config)
        config_df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'config_analysis.csv'), index=False)
        flash("✅ PROC configuration saved successfully!", "success")
        return redirect('/config-visual')

    return render_template(
        'config-analysis.html',
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        binary_columns=binary_columns,
        df=df  # needed for SGPLOT dropdown
    )


# SAS Visualization Procs
VISUAL_PROCS = {
    "PROC SGPLOT": {
        "desc": "Simple bar, line, scatter, or histogram",
        "roles": ["x", "y", "group"]
    },
    "PROC SGSCATTER": {
        "desc": "Scatter plot matrix of numeric variables",
        "roles": ["variables"]
    },
    "PROC SGPANEL": {
        "desc": "Faceted (small multiple) plots",
        "roles": ["x", "y", "panel"]
    },
    "PROC BOXPLOT": {
        "desc": "Boxplot for grouped distributions",
        "roles": ["category", "y"]
    },
    "PROC GCHART": {
        "desc": "Bar/Pie chart using categories",
        "roles": ["category", "response"]
    },
    "PROC GCONTOUR": {
        "desc": "Contour plot",
        "roles": ["x", "y", "z"]
    },
    "PROC G3D": {
        "desc": "3D scatter or surface plot",
        "roles": ["x", "y", "z"]
    }
}

@app.route('/config-visual', methods=['GET', 'POST'])
def config_visual():
    sampled_path = os.path.join(app.config['UPLOAD_FOLDER'], 'sampled_data.pkl')
    if not os.path.exists(sampled_path):
        flash("⚠️ Please sample the dataset first.", "warning")
        return redirect('/sample')

    df = pd.read_pickle(sampled_path)
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    if request.method == 'POST':
        visual_config = []

        # MEANS
        for col in request.form.getlist('means_visual_columns'):
            visual_config.append({'PROC': 'MEANS', 'Visual': f'Box/Histogram for {col}'})

        # FREQ
        for col in request.form.getlist('freq_visual_columns'):
            visual_config.append({'PROC': 'FREQ', 'Visual': f'Bar/Pie chart for {col}'})

        # LOGISTIC (default visual placeholder)
        visual_config.append({'PROC': 'LOGISTIC', 'Visual': 'ROC Curve / Accuracy plot'})

        # SQL (no fixed visual)
        visual_config.append({'PROC': 'SQL', 'Visual': 'Depends on query output'})

        # CORR
        for col in request.form.getlist('corr_visual_columns'):
            visual_config.append({'PROC': 'CORR', 'Visual': f'Heatmap for {col}'})

        # REG
        reg_x = request.form.get('reg_x')
        reg_y = request.form.get('reg_y')
        if reg_x and reg_y:
            visual_config.append({'PROC': 'REG', 'Visual': f'Regression plot X={reg_x}, Y={reg_y}'})

        # UNIVARIATE
        for col in request.form.getlist('univariate_visual_columns'):
            visual_config.append({'PROC': 'UNIVARIATE', 'Visual': f'Skew/Kurtosis for {col}'})

        # SGPLOT
        sgplot_x = request.form.get('sgplot_x')
        sgplot_y = request.form.get('sgplot_y')
        if sgplot_x and sgplot_y:
            visual_config.append({'PROC': 'SGPLOT', 'Visual': f'Custom plot: X={sgplot_x}, Y={sgplot_y}'})

        # Save the visual configuration
        config_df = pd.DataFrame(visual_config)
        config_df.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'config_visual.csv'), index=False)
        flash("✅ Visualization configuration saved!", "success")

        # Optional: render preview summary
        preview_html = config_df.to_html(classes="table table-bordered table-sm", index=False)
        return render_template(
            'config-visual.html',
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            df=df,
            preview_gallery=preview_html
        )

    return render_template(
        'config-visual.html',
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        df=df,
        preview_gallery=None
    )

@app.route('/walkthrough')
def walkthrough():
    return render_template('walkthrough.html')


if __name__ == "__main__":
    app.run(debug=True)
