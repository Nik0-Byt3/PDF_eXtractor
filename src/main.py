from utils.PDFReader import PDFExtractor
from utils.llm_client import LLMClient
import os 
import argparse
import logging
import time 
from config import BaseConfig as Config
from utils.app import App
import utils.views


class PDFProcessor(object):

    def __init__(self, pdf_path , output_path=None):
        self.pdf_extractor = PDFExtractor(pdf_path)
        self.book_info = self.pdf_extractor.get_book_info()
        self.output_path = os.path.join(output_path , self.book_info['title'].replace(" " , "-").replace("," , "-").replace(":" , "-"))
        self.pdf_extractor.set_path(self.output_path)

    

    def process_pdf(self):
        
        print(f"Author: {self.book_info['author']}")
        print(f"Title: {self.book_info['title']}")
        print(f"Number of Pages: {self.book_info['num_pages']}")
        self.content = self.pdf_extractor.get_cleaned_text()

    def send_to_llm(self):
        self.llm = LLMClient(self.content , self.output_path)
        self.llm.main()
        
    def get_chapters_path(self):
        return self.pdf_extractor.chapter_dir


'''parser = argparse.ArgumentParser( prog='main.py', description='Process PDF files and extract information')
parser.add_argument('input_path', type=str, help='Path to the PDF file')
parser.add_argument('--output-dir', type=str, help='Path to the output directory')

args = parser.parse_args()'''

logger = logging.getLogger(__name__)
FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(
    format=FORMAT,
    filename='PDF_CURSER_GENERATOR.log',
    level=Config.LOG_LEVEL,
)
logger.info('Started')

App.run()

'''t = time.time()
Book = PDFProcessor(args.input_path, args.output_dir)

Book.pdf_extractor.run_chapter_recognition_test() 
Book.pdf_extractor.extract_images() if Config.ENABLE_IMAGE_EXTRACTION else None
chap_dir = sorted_files = sorted(os.listdir(Book.get_chapters_path()), key=lambda x: int(x.split('_')[1].split('.')[0]))

for i , chapter_pdf in enumerate(chap_dir):
    chapter_pdf_path = os.path.join(Book.get_chapters_path(), chapter_pdf)
    logger.info(f'Processing chapter: {chapter_pdf}')
    chapter_processor = PDFProcessor(chapter_pdf_path , os.path.join(args.output_dir , f'chapter-{i+1}'))
    chapter_processor.process_pdf()
    chapter_processor.send_to_llm()

logger.info('Finished processing PDF file')
elapsed_time = (time.time() - t) / 60
logger.info(f'Elapsed time: {elapsed_time:.2f} minutes')'''
