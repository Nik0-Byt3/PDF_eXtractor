from utils.app import App
from flask import render_template, request, jsonify , Response , stream_with_context, redirect, url_for
from utils.PDFReader import PDFExtractor
from utils.llm_client import LLMClient
from utils.PDFProcessor import PDFProcessor
import os
from config import BaseConfig as Config
import tempfile
import shutil
import tempfile
import PyPDF2
import time
import json
import shutil
from utils.logger import log_to_client , clear_log
import uuid
from flask import send_file
from utils.Pdf_converter import BookMarkdownToPPTX



@App.route('/')
def root():
    return render_template("homepage.html")




@App.route('/chapters', methods=['POST'])
def list_chapters():
    user_file = request.files.get('pdfFile')
    if not user_file:
        return jsonify({'error': 'No file uploaded'}), 400

    temp_path = os.path.join(tempfile.gettempdir(), user_file.filename)
    user_file.save(temp_path)

    extractor = PDFExtractor(temp_path)
    
    try:
        with open(temp_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            outline = reader.outline
            if not outline:
                return jsonify({'error': 'Nessun indice capitoli trovato nel PDF'}), 404
            outline_info = extractor.get_pdf_outline_info(outline, reader)
            page_ranges = extractor.calculate_page_ranges(outline_info, len(reader.pages))
            return jsonify({'sections': page_ranges, 'pdf_path': temp_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@App.route('/extract_chapters', methods=['POST'])
def extract_chapters():
    selected = request.json.get('selected_indices')
    pdf_path = request.json.get('pdf_path')

    if not selected or not pdf_path:
        return jsonify({'error': 'Missing data'}), 400

    try:
        # Estrai capitoli scelti
        extractor = PDFExtractor(pdf_path)
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            outline = reader.outline
            outline_info = extractor.get_pdf_outline_info(outline, reader)
            page_ranges = extractor.calculate_page_ranges(outline_info, len(reader.pages))

            # Seleziona solo i capitoli richiesti
            selected_sections = [page_ranges[i] for i in selected]
            extractor.save_sections_as_pdfs(pdf_path, selected_sections)

        chap_dir = extractor.chapter_dir

        all_files = sorted(
            [f for f in os.listdir(chap_dir) if f.endswith('.pdf')],
            key=lambda x: int(x.split("_")[0])  
        )
        for i, chapter_pdf in enumerate(all_files):
            log_to_client(f"Elaborating selected chapter {selected[i]+1}: {chapter_pdf}")
            chapter_pdf_path = os.path.join(chap_dir, chapter_pdf)

            # Usa il nome del file PDF
            pdf_basename = os.path.splitext(os.path.basename(chapter_pdf))[0]
            chapter_out_dir = os.path.join(chap_dir, f'{pdf_basename}')

            os.makedirs(chapter_out_dir, exist_ok=True)

            chapter_processor = PDFProcessor(chapter_pdf_path, chapter_out_dir)
            chapter_processor.get_images()
            chapter_processor.process_pdf()
            chapter_processor.send_to_llm()


        clear_log()
        
        converter = BookMarkdownToPPTX(chap_dir, chap_dir)
        converter.convert_all_chapters()

        # Compressione ZIP
        zip_filename = f"{uuid.uuid4()}.zip"
        zip_path = os.path.join(tempfile.gettempdir(), zip_filename)
        shutil.make_archive(zip_path.replace('.zip', ''), 'zip', chap_dir)
        shutil.rmtree(chap_dir)

        if not os.path.isfile(zip_path):
            log_to_client(f"ERRORE: file zip non trovato: {zip_path}")
        else:
            log_to_client(f"File zip trovato: {zip_path}")

        log_to_client(f"Chapters extracted and saved to {zip_path}")
        return jsonify({'zip_file': zip_filename, 'chapters': all_files})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@App.route('/settings')
def settings():
    return render_template("settings.html")

# Define user_config as a global dictionary to store user settings
user_config = {
    "ENABLE_TRANSLATION": False,
    "ENABLE_SUMMARIZATION": False,
    "ENABLE_IMAGE_EXTRACTION": False
}

@App.route("/impostazioni", methods=["GET"])
def mostra_impostazioni():
    return render_template("settings.html", config=user_config)

@App.route("/salva_impostazioni", methods=["POST"])
def salva_impostazioni():
    os.environ["ENABLE_TRANSLATION"] = "True" if "traduzione" in request.form else "False"
    os.environ["ENABLE_SUMMARIZATION"] = "True" if "riassunto" in request.form else "False"
    os.environ["ENABLE_IMAGE_EXTRACTION"] = "True" if "estrazione_immagini" in request.form else "False"

    print("TRADUZIONE:", os.environ.get("ENABLE_TRANSLATION"))
    print("RIASSUNTO:", os.environ.get("ENABLE_SUMMARIZATION"))
    print("ESTRAZIONE IMMAGINI:", os.environ.get("ENABLE_IMAGE_EXTRACTION"))
    print("="*10)
    # Aggiorna il dizionario user_config con i valori delle impostazioni
    user_config["ENABLE_TRANSLATION"] = os.environ.get("ENABLE_TRANSLATION")
    user_config["ENABLE_SUMMARIZATION"] = os.environ.get("ENABLE_SUMMARIZATION")
    user_config["ENABLE_IMAGE_EXTRACTION"] = os.environ.get("ENABLE_IMAGE_EXTRACTION")

    print("Config salvata:", user_config)

    return redirect("/")


@App.route("/Download_file")
def download_file():
    zip_filename = request.args.get('file')
    if not zip_filename:
        return jsonify({'error': 'Missing file parameter'}), 400

    # Costruisci il percorso completo nella cartella temporanea
    zip_path = os.path.join(tempfile.gettempdir(), zip_filename)

    if os.path.isfile(zip_path):
        log_to_client(f"Download file trovato: {zip_path}")
        return send_file(zip_path, as_attachment=True, mimetype='application/zip')
    else:
        log_to_client(f"File zip non trovato al momento del download: {zip_path}")
        return jsonify({'error': 'File not found'}), 404
    
@App.route('/clear')
def clear():
    zipfile = request.args.get('zipfile')
    if zipfile:
        zip_path = os.path.join(tempfile.gettempdir(), zipfile)
        if os.path.isfile(zip_path):
            try:
                os.remove(zip_path)
                log_to_client(f"File zip {zipfile} rimosso con successo.")
            except Exception as e:
                log_to_client(f"Errore durante la rimozione del file zip: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        else:
            log_to_client(f"File zip {zipfile} non trovato.")
    try:
        log_to_client("Temporary files cleared successfully.")
    except Exception as e:
        log_to_client(f"Error clearing temporary files: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    clear_log()
    log_to_client("Log file cleared successfully.")
    
    return jsonify({'status': 'success', 'message': 'Data cleared successfully.'})
