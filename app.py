import os
import pdfplumber
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import csv
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Configuração básica do logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Função para carregar os nomes dos arquivos já processados
def load_processed_files(progress_file):
    processed_files = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    processed_files.add(row[0])
    return processed_files

# Função para salvar o progresso dos arquivos processados
def save_processed_file(progress_file, filename):
    with open(progress_file, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([filename])

# Função para extrair texto de PDFs com tqdm
def extract_text_from_pdfs(directory, output_file, progress_file):
    logging.info(f"Começando a extração de texto dos PDFs no diretório: {directory}")
    
    pdf_files = [f for f in os.listdir(directory) if f.endswith(".pdf")]
    processed_files = load_processed_files(progress_file)
    new_files = [f for f in pdf_files if f not in processed_files]
    
    logging.info(f"Encontrados {len(new_files)} novos arquivos PDF para processar.")
    
    with open(output_file, 'a', encoding='utf-8') as text_file:
        for filename in tqdm(new_files, desc="Processando PDFs", unit="pdf", ncols=100):
            pdf_path = os.path.join(directory, filename)
            logging.info(f"Processando arquivo PDF: {filename}")
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in tqdm(pdf.pages, desc=f"Processando {filename}", unit="página", leave=False, ncols=100):
                        text = page.extract_text()
                        if text:  # Verifique se há texto extraído
                            text_file.write(text + "\n")
                        else:
                            logging.warning(f"Nenhum texto extraído da página do arquivo PDF: {filename}")
            except Exception as e:
                logging.error(f"Erro ao processar o arquivo {filename}: {e}")
    
            # Marcar o arquivo como processado
            save_processed_file(progress_file, filename)

# Função para carregar o texto e vetorizar em chunks menores
def load_text_and_vectorize(file_path, chunk_size=1000):
    logging.info(f"Lendo texto do arquivo: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Divida o texto em chunks menores baseados no tamanho máximo
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    # Vetorização TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)
    
    return tfidf_matrix, vectorizer, chunks

# Função para obter a resposta mais relevante para a pergunta
def get_dnd_response(question, tfidf_matrix, vectorizer, chunks):
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, tfidf_matrix).flatten()
    
    # Filtrar chunks que tenham uma similaridade mínima
    threshold = 0.1  # Ajuste esse valor conforme necessário
    best_chunk_idx = np.argmax(similarities)
    
    if similarities[best_chunk_idx] > threshold:
        return chunks[best_chunk_idx]
    else:
        return "Sem respostas para essa pergunta."

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://127.0.0.1:5500"}})

@app.route('/get-dnd-response', methods=['POST'])
def api_get_dnd_response():
    """Rota de API para obter uma resposta para uma pergunta sobre D&D."""

    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "A pergunta é necessária."}), 400

    try:
        # Carregar e vetorização do texto
        tfidf_matrix, vectorizer, chunks = load_text_and_vectorize("extracted_text.txt")
        
        # Obter resposta para a pergunta
        answer = get_dnd_response(question, tfidf_matrix, vectorizer, chunks)
    except Exception as e:
        logging.error(f"Erro ao processar a pergunta: {e}")
        return jsonify({"error": "Ocorreu um erro ao processar sua pergunta."}), 500

    return jsonify({"answer": answer})

if __name__ == '__main__':
    # Executar a aplicação Flask
    logging.info("Iniciando o servidor Flask.")
    app.run(debug=True)
