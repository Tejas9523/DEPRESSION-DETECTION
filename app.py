from flask import Flask, render_template, request, render_template_string
import os
import be  # Ensure your be.py has a main(video_path) function

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max

# Ensure uploads directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/run-assessment", methods=["POST"])
def run_assessment():
    uploaded_file = request.files.get("videoFile")

    if uploaded_file and uploaded_file.filename != "":
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(filepath)

        # Call your processing function with the video path
        report_html = be.main(filepath)

        return render_template_string(report_html)
    else:
        return "No file uploaded", 400

if __name__ == "__main__":
    app.run(debug=True)
