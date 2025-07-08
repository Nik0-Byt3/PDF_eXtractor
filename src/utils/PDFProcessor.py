from utils.PDFReader import PDFExtractor
from utils.llm_client import LLMClient
from utils.logger import log_to_client
import os
import tempfile
import PyPDF2
import asyncio
from concurrent.futures import ThreadPoolExecutor



class PDFProcessor(object):



    def __init__(self, pdf_path, output_path=None):
        self.pdf_extractor = PDFExtractor(pdf_path)
        self.book_info = self.pdf_extractor.get_book_info()
        
        if output_path is None:
            output_path = tempfile.gettempdir()

        title_safe = self.book_info['title'].replace(" ", "-").replace(",", "-").replace(":", "-") if self.book_info['title'] else None
        if title_safe is None:
            self.output_path = output_path
        else:
            self.output_path = os.path.join(output_path, title_safe)
        self.pdf_extractor.set_path(self.output_path)


    def process_pdf(self):
        
        print(f"Author: {self.book_info['author']}")
        print(f"Title: {self.book_info['title']}")
        print(f"Number of Pages: {self.book_info['num_pages']}")
        self.content =  self.pdf_extractor.get_cleaned_text()

    def send_to_llm(self):
        self.llm = LLMClient(self.content , self.output_path)
        self.llm.main()
        
    def get_chapters_path(self):
        if not self.pdf_extractor.chapter_dir:
            raise AttributeError("Chapter directory is not initialized. Did you run chapter recognition?")
        return self.pdf_extractor.chapter_dir
    
    def extract_chapters_by_indices(self, selected_indices):
        with open(self.pdf_extractor.pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            outline = reader.outline
            outline_info = self.pdf_extractor.get_pdf_outline_info(outline, reader)
            page_ranges = self.pdf_extractor.calculate_page_ranges(outline_info, len(reader.pages))
            selected_sections = [page_ranges[i] for i in selected_indices]
            self.pdf_extractor.save_sections_as_pdfs(self.pdf_extractor.pdf_path, selected_sections)

    def get_images(self):
        immages = asyncio.run(self.pdf_extractor.extract_images())
        if not immages:
            log_to_client("No images found in the PDF.")
        else:
            log_to_client(f"Extracted {len(immages)} images from the PDF.")
        return immages

            
