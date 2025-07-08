import os
import re
import fitz  # PyMuPDF
import pdfplumber
import logging
from sentence_transformers import SentenceTransformer
import PyPDF2
from PyPDF2 import PdfReader, PdfWriter
from utils.logger import log_to_client
import asyncio
from concurrent.futures import ThreadPoolExecutor
import unicodedata

logger = logging.getLogger(__name__)

class PDFExtractor:
    def __init__(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")
        self.pdf_path = pdf_path
        self.pdf = pdfplumber.open(pdf_path)
        self.num_pages = len(self.pdf.pages)
        self.doc = fitz.open(pdf_path)
        self.output_path = "src/output"
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chapter_dir = None

    def set_path(self, output_path):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)


    def get_book_info(self):
        return {
            "author": self.pdf.metadata.get("Author", "Unknown"),
            "title": self.pdf.metadata.get("Title", "Unknown"),
            "num_pages": self.num_pages
        }

    def get_cleaned_text(self, texts=None):
        if texts is None:
            logger.info("Starting text extraction...")
            log_to_client("Starting text extraction...")

            def extract_text(page):
                return page.get_text()

            texts = [extract_text(page) for page in self.doc]

            output_file = os.path.join(self.output_path, "cleaned_output.md")

            with open(output_file, "w") as f:
                for number, text in enumerate(texts):
                    if text:
                        f.write(text)
                    f.write("\n\n")

        logger.info(f"Text extraction completed and saved to {output_file}")
        log_to_client(f"Text extraction completed and saved to {output_file}")
        return texts


    def extract_caption(self, page, img_bbox):
        blocks = page.get_text("blocks")
        caption = None
        min_dist = float("inf")
        for block in blocks:
            x0, y0, x1, y1, text, *_ = block
            if y0 >= img_bbox[3]:
                dist = y0 - img_bbox[3]
                if dist < min_dist and text.strip():
                    min_dist = dist
                    caption = text.strip().replace('\n', ' ')
        return caption

    async def extract_images(self):
        logger.info("Starting image extraction...")
        log_to_client("Starting image extraction...")
        extract = []
        image_count = 0
        images_dir = os.path.join(self.output_path, "images")
        os.makedirs(images_dir, exist_ok=True)

        loop = asyncio.get_running_loop()
        executor = ThreadPoolExecutor()

        async def process_page(page_num):
            page_extract = []
            page = self.doc.load_page(page_num)
            images = page.get_images(full=True)

            for img_index, img in enumerate(images):
                xref = img[0]
                pix = fitz.Pixmap(self.doc, xref)
                if pix.n - pix.alpha == 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                img_bbox = page.get_image_bbox(img)
                caption = self.extract_caption(page, img_bbox)
                safe_caption = "".join(c for c in (caption or "") if c.isalnum() or c in (' ', '_', '-')).strip()
                img_filename = f"{safe_caption or f'image_page{page_num + 1}_{img_index + 1}'}.png"
                img_path = os.path.join(images_dir, img_filename)

                pix.save(img_path)
                pix = None  # free memory

                figure_info = {
                    "filename": img_filename,
                    "path": img_path,
                    "page": page_num + 1,
                    "index": img_index + 1,
                    "size": os.path.getsize(img_path) if os.path.exists(img_path) else 0
                }
                page_extract.append(figure_info)

            return page_extract

        tasks = [process_page(pn) for pn in range(self.doc.page_count)]
        results = await asyncio.gather(*tasks)

        for page_results in results:
            extract.extend(page_results)
            image_count += len(page_results)

        logger.info(f"Image extraction completed. Total images extracted: {image_count}")
        log_to_client(f"Image extraction completed. Total images extracted: {image_count}")
        return extract

    def get_pdf_outline_info(self, outline, reader, level=1, parent_numbers=[]):
        outline_info = []
        counter = 1

        for item in outline:
            if isinstance(item, PyPDF2.generic.Destination):
                title = item.title.strip()
                page_index = reader.get_page_number(item.page)
                numbering = ".".join(map(str, parent_numbers + [counter]))
                outline_info.append({
                    "title": title,
                    "page_index": page_index,
                    "level": level,
                    "numbering": numbering
                })
                # Se il prossimo elemento è una lista, esplora ricorsivamente
                
                next_index = outline.index(item) + 1
                if level < 2:
                    if next_index < len(outline) and isinstance(outline[next_index], list):
                        sub_outline = self.get_pdf_outline_info(
                            outline[next_index], reader, level + 1, parent_numbers + [counter]
                        )
                        outline_info.extend(sub_outline)
                counter += 1

        return outline_info



    def calculate_page_ranges(self, outline_info, total_pages):
        sections = []
        outline_info.sort(key=lambda x: x["page_index"])

        for i, item in enumerate(outline_info):
            start_index = item["page_index"]
            end_index = outline_info[i + 1]["page_index"] - 1 if i + 1 < len(outline_info) else total_pages - 1

            if start_index <= end_index:
                sections.append({
                    "name": f"{item['numbering']} {item['title']}",
                    "start_page": start_index + 1,
                    "end_page": end_index + 1
                })

        return sections


    

    def save_sections_as_pdfs(self, pdf_path, sections):
        reader = PdfReader(pdf_path)
        output_dir = os.path.join(self.output_path, "chapters")
        self.chapter_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        for i, section in enumerate(sections):
            writer = PdfWriter()
            for page_num in range(section['start_page'] - 1, section['end_page']):
                writer.add_page(reader.pages[page_num])

            sanitized_title = self.sanitize_filename(section['name'])
            filename = f"{i}_{sanitized_title}.pdf"
            output_path = os.path.join(output_dir, filename)
            
            with open(output_path, "wb") as f:
                writer.write(f)

            logger.info(f"✅ Saved: {output_path}")
            
            
    def sanitize_filename(self , name):
        
        name = unicodedata.normalize("NFKD", name)
        name = re.sub(r"[^\w\s-]", "", name)  
        name = re.sub(r"\s+", "_", name.strip())  
        # remove all numbers
        name = re.sub(r"\d+", "", name)  
        return name[:100]  


    def run_chapter_recognition_test(self, selected_indices=None):
        try:
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                logger.info(f"Total pages in PDF: {total_pages}")

                outline = reader.outline
                if not outline:
                    logger.info("No bookmarks (outline) found in PDF.")
                    return

                outline_info = self.get_pdf_outline_info(outline, reader)
                outline_info = [item for item in outline_info if item["page_index"] is not None]

                if not outline_info:
                    logger.info("No valid bookmark entries found.")
                    return

                page_ranges = self.calculate_page_ranges(outline_info, total_pages)

                logger.info("\nAvailable chapters and subchapters:")
                for i, section in enumerate(page_ranges):
                    logger.info(f"{i + 1}. {section['name']} (Pages: {section['start_page']}-{section['end_page']})")

                if selected_indices is None:
                    logger.info("No chapter selection provided.")
                    return

                selected_sections = [page_ranges[i] for i in sorted(set(selected_indices)) if 0 <= i < len(page_ranges)]
                if not selected_sections:
                    logger.info("No valid selection made. Exiting.")
                    return

                self.save_sections_as_pdfs(self.pdf_path, selected_sections)

        except Exception as e:
            logger.error(f"Error while processing PDF: {e}")
