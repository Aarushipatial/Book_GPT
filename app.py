from flask import Flask, request, render_template
import fitz  # PyMuPDF for PDF handling
from transformers import pipeline
import os

app = Flask(__name__)

# Load the question-answering model
question_answerer = pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def preprocess_text(text):
    text = text.lower().strip()
    return text

def get_answer(question, context):
    result = question_answerer(question=question, context=context)
    return result['answer']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        question = request.form['question']
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        text = extract_text_from_pdf(file_path)
        text = preprocess_text(text)
        answer = get_answer(question, text)
        
        return render_template('index.html', answer=answer)

    return render_template('index.html', answer="")

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
