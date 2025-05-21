from flask import Flask, request, render_template, redirect, url_for, jsonify, flash, send_file
import os
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
from werkzeug.utils import secure_filename
from resume_model import ResumeProcessor, ResumeModel

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs("model", exist_ok=True)

processor = ResumeProcessor()
model = ResumeModel()

@app.context_processor
def inject_now():
    return {'now': datetime.now()}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_files():
    if 'resumes' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    files = request.files.getlist('resumes')
    
    if not files or files[0].filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            uploaded_files.append(filename)
        else:
            flash(f'Invalid file type: {file.filename}. Only PDF files are allowed.', 'error')

    if uploaded_files:
        flash(f'Successfully uploaded {len(uploaded_files)} resume(s)', 'success')
        try:
            resume_df = processor.process_resumes()
            resume_df.to_json("parsed_resumes.json", orient="records", indent=4)
            flash('Resumes processed successfully', 'success')
        except Exception as e:
            flash(f'Error processing resumes: {str(e)}', 'error')

    return redirect(url_for('index'))

@app.route("/match", methods=["POST"])
def match_resumes():
    job_description = request.form.get("job_description", "").strip()
    
    if not job_description:
        return jsonify({"error": "Job description is required"}), 400

    try:
        with open("parsed_resumes.json", 'r') as f:
            resumes_data = json.load(f)
        
        if not resumes_data:
            return jsonify({"error": "No processed resumes found"}), 400

        job_embedding = model.encode_resume(job_description)
        
        results = []
        for resume in resumes_data:
            resume_embedding = model.encode_resume(resume['Resume'])
            similarity = float(np.dot(job_embedding, resume_embedding) / 
                            (np.linalg.norm(job_embedding) * np.linalg.norm(resume_embedding)))
            
            results.append({
                "filename": resume['filename'],
                "similarity": similarity,
                "skills": resume['Skills'],
                "experience": resume['Experience'],
                "education": resume['Education']
            })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return render_template(
            "results.html",
            matches=results,
            job_description=job_description
        )

    except Exception as e:
        flash(f'Error matching resumes: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route("/analytics")
def analytics():
    try:
        with open("parsed_resumes.json", 'r') as f:
            resumes_data = json.load(f)
        
        total_resumes = len(resumes_data)
        skills_distribution = {}
        experience_levels = {"0-2": 0, "3-5": 0, "5+": 0}
        
        for resume in resumes_data:
            skills = resume.get('Skills', '').lower().split()
            for skill in skills:
                skills_distribution[skill] = skills_distribution.get(skill, 0) + 1
            
            exp_text = resume.get('Experience', '').lower()
            years = sum([int(y) for y in exp_text.split() if y.isdigit() and int(y) < 50])
            if years <= 2:
                experience_levels["0-2"] += 1
            elif years <= 5:
                experience_levels["3-5"] += 1
            else:
                experience_levels["5+"] += 1

        return render_template(
            "analytics.html",
            total_resumes=total_resumes,
            skills_distribution=dict(sorted(skills_distribution.items(), 
                                         key=lambda x: x[1], reverse=True)[:10]),
            experience_levels=experience_levels
        )

    except Exception as e:
        flash(f'Error loading analytics: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route("/export")
def export_results():
    try:
        with open("parsed_resumes.json", 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        output_file = f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        df.to_excel(output_file, index=False)
        
        return send_file(
            output_file,
            as_attachment=True,
            download_name=output_file
        )

    except Exception as e:
        flash(f'Error exporting results: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    app.run(debug=True)
