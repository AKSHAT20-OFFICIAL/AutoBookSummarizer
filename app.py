import os
import io
import base64
from flask import Flask, request, jsonify, render_template, send_file
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader, PdfWriter
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHUNK_FOLDER'] = 'uploads/chunks'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure the upload and chunk folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHUNK_FOLDER'], exist_ok=True)

# Check if the file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Function to split the PDF into chunks of 10 pages and save chunks in CHUNK_FOLDER
def split_pdf_into_chunks(file_path, chunk_folder, chunk_size=10):
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    chunks = []

    for i in range(0, total_pages, chunk_size):
        writer = PdfWriter()
        for j in range(i, min(i + chunk_size, total_pages)):
            writer.add_page(reader.pages[j])

        # Save the chunk to a file
        chunk_filename = f"chunk_{i // chunk_size + 1}.pdf"
        chunk_path = os.path.join(chunk_folder, chunk_filename)
        with open(chunk_path, 'wb') as chunk_file:
            writer.write(chunk_file)
        chunks.append(chunk_path)

    return chunks

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Save the uploaded file
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"Uploaded file saved at: {file_path}")

        try:
            # Split the PDF into chunks
            chunks = split_pdf_into_chunks(file_path, app.config['CHUNK_FOLDER'])
            responses = []

            # Initialize the generative model
            model = genai.GenerativeModel("gemini-1.5-flash")

            # Process each chunk
            for index, chunk_path in enumerate(chunks):
                # Define prompt based on whether it's the last chunk
                prompt = (
                    "Using all the information you have gathered, write a review of this book as a book critique."
                    if index == len(chunks) - 1
                    else "Read and understand this document."
                )

                # Read and encode the chunk file
                with open(chunk_path, "rb") as chunk_file:
                    chunk_data = base64.standard_b64encode(chunk_file.read()).decode('utf-8')

                # Send the chunk to the API
                response = model.generate_content(
                    [
                        {
                            "mime_type": "application/pdf",
                            "data": chunk_data
                        },
                        prompt
                    ]
                )

                # Access the text attribute of the response object
                responses.append(response.text)

            # Combine and return the final review
            final_review = "\n".join(responses)

            # Send the review as a downloadable file
            review_file = io.BytesIO(final_review.encode('utf-8'))
            review_file.seek(0)

            return send_file(
                review_file,
                mimetype='text/plain',
                as_attachment=True,
                download_name='review.txt',
            )

        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({'error': str(e)}), 500

        finally:
            # Clean up the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)
