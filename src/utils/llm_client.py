from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

import logging
import os 
from  transformers import AutoTokenizer, AutoModel , pipeline , AutoModelForSeq2SeqLM
import torch 
import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sentencepiece
import re
import sacremoses
import textwrap
import nltk
import textwrap
from nltk.tokenize import TextTilingTokenizer, sent_tokenize
import unicodedata
from datasets import Dataset
import gc
from config import BaseConfig as Config
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import math
from utils.logger import log_to_client



logger = logging.getLogger(__name__)



class LLMClient:

    def __init__(self, content, output_path):
        self.device = Config.DEVICE    
        logger.info(f"Using device: {self.device}")
        if Config.DEVICE == "cuda" and  torch.cuda.is_available():
            logger.debug(f"CUDA is available. GPU details:")
            logger.debug(f"GPU Name: {torch.cuda.get_device_name(0)}")
            logger.debug(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            logger.debug(f"CUDA Memory Cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
            logger.debug(f"Total CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB")
            logger.debug(f"CUDA Version: {torch.version.cuda}")
        self.content = content
        self.output_path = output_path
        self.translator = None
        os.environ['TRANSFORMERS_CACHE'] = 'src/AI/RAG_models'
        os.environ['TRITON_CACHE_DIR'] = 'src/AI/RAG_models'
        os.environ['CUDA_LAUNCH_BLOCKING']="1"
        os.environ['TORCH_USE_CUDA_DSA'] = "1"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.translation_lock = threading.Lock()
        self.translated_count = 0

        nltk.download('punkt')

    

    def main(self):
        self.db = chromadb.PersistentClient(path="src/AI/chroma_db" , settings=Settings(anonymized_telemetry=False))
        self.detect_language()
        self.chunk_content()
        t = asyncio.run(self.translate())
        self.load_model()
        self.calculate_embeddings()
        self.load_in_chromadb(os.path.basename(self.output_path))
        logger.info(f"Data loaded into ChromaDB at src/AI/chroma_db with collection name {os.path.basename(self.output_path)}")
        self.summarize()
        self.unify_summarized_text()
        logger.info("LLMClient processing completed successfully.")
        log_to_client("LLMClient processing completed successfully.")




    def detect_language(self):
        with open(f"{self.output_path}/cleaned_output.md" , "r", encoding="utf-8") as file:
            self.text = sanitize_text(file.read())

        try:
            self.language = detect(self.text)
            logger.info(f"Detected language: {self.language}")
            return self.language
        except LangDetectException as e:
            logger.error(f"Language detection failed: {e}")
            return None

    async def async_translate_batches(self, dataset, tokenizer, model, batch_size, valid_indices):
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=4)
        translated_chunks = [""] * len(self.chunks)

        async def translate_batch(start):
            batch = dataset[start:start + batch_size]

            input_ids = torch.tensor(batch["input_ids"], dtype=torch.long).to(self.device)
            attention_mask = torch.tensor(batch["attention_mask"], dtype=torch.long).to(self.device)
            self.cuda_cleanup()  # dopo allocazione tensori

            def generate_output():
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=512,
                        num_beams=6,
                        no_repeat_ngram_size=3,
                        length_penalty=1.1,
                        repetition_penalty=2.5,
                        early_stopping=True
                    )
                    return outputs

            outputs = await loop.run_in_executor(executor, generate_output)
            self.cuda_cleanup()  # subito dopo generate()

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            self.cuda_cleanup()  # dopo decoding

            
            for j, translation in enumerate(decoded):
                idx = valid_indices[start + j]
                translated_chunks[idx] = translation

                with self.translation_lock:
                    self.translated_count += 1
                    progress = self.translated_count / len(self.chunks)
                    bar_length = 40
                    filled_length = int(round(bar_length * progress))
                    bar = '█' * filled_length + '-' * (bar_length - filled_length)
                    percent = math.ceil(progress * 100)


                    chunk_interval = max(1, len(self.chunks) // 20)  
                    if (
                        self.translated_count == 1 or
                        self.translated_count % chunk_interval == 0 or
                        self.translated_count == len(self.chunks)
                    ):
                        logger.info(f"[{bar}] {percent:5.1f}% - Translated {self.translated_count}/{len(self.chunks)}")
                        log_to_client(f"[{bar}] {percent:5.1f}% - Translated {self.translated_count}/{len(self.chunks)}")




            # cleanup finale di sicurezza
            del input_ids
            del attention_mask
            del outputs
            self.cuda_cleanup()

        tasks = [translate_batch(start) for start in range(0, len(dataset), batch_size)]
        await asyncio.gather(*tasks)

        self.cuda_cleanup()  # cleanup globale a fine async
        return translated_chunks

    
    async def translate(self):
        if self.language == "it":
            logger.info("No translation needed, text is already in Italian.")
            return

        try:
            model_name = f"Helsinki-NLP/opus-mt-{self.language}-it"
            logger.info(f"Initializing translation model: {model_name}")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            except Exception as e:
                logger.error(f"Model '{model_name}' not available: {e}")
                log_to_client(f"Translation model for '{self.language} → it' not found.")
                return

            logger.info("Sanitizing and preparing input text...")
            data = []
            valid_indices = []

            for i, raw_text in enumerate(self.chunks):
                clean_text = sanitize_text(raw_text)

                if not clean_text.strip():
                    logger.warning(f"Skipping empty/invalid chunk {i + 1}")
                    continue

                if len(clean_text) > 8000:
                    logger.warning(f"Truncating chunk {i + 1} to 8000 characters")
                    clean_text = clean_text[:8000]

                data.append({"id": i, "text": clean_text})
                valid_indices.append(i)

            if not data:
                logger.warning("No valid chunks to translate.")
                return

            dataset = Dataset.from_list(data)

            def tokenize(batch):
                return tokenizer(
                    batch["text"],
                    padding="longest",
                    truncation=True,
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt"
                )

            logger.info("Tokenizing input for translation...")
            dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

            logger.info(f"Starting async translation of {len(dataset)} chunks...")
            log_to_client(f"Starting async translation of {len(dataset)} chunks...")

            translated_chunks = await self.async_translate_batches(
                dataset, tokenizer, model, batch_size=4, valid_indices=valid_indices
            )

            # Clean output
            self.chunks = [sanitize_text(t) for t in translated_chunks if t.strip()]
            wrapped_chunks = [textwrap.fill(c, width=120) for c in self.chunks]
            self.text = "\n".join(wrapped_chunks)

            output_file = f"{self.output_path}/translated_output.md"
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(self.text)

            logger.info("All translations completed successfully.")
            log_to_client("Translation completed.")

        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            self.cuda_cleanup()







    def fix_and_write_text(self, output_path="output.txt", wrap_width=80):
        """
        Reconstructs paragraph structure using TextTiling + NLTK tokenization,
        wraps text to given width, and writes to file.
        """

        raw = self.text

        import re
        cleaned = re.sub(r'(?<![.?!])\n(?!\n)', ' ', raw)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        tt = TextTilingTokenizer()
        try:
            paragraphs = tt.tokenize(cleaned)
        except ValueError:

            sentences = sent_tokenize(cleaned)
            paragraphs = [
                ' '.join(sentences[i:i+5]) for i in range(0, len(sentences), 5)
            ]
        wrapper = textwrap.TextWrapper(width=wrap_width)
        wrapped_paragraphs = [wrapper.fill(p.strip()) for p in paragraphs if p.strip()]
        with open(output_path, 'w', encoding='utf-8') as f:
            for para in wrapped_paragraphs:
                f.write(para + "\n\n") 



    def chunk_content(self, chunk_size=800, chunk_overlap=150):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.chunks = text_splitter.split_text(self.text)
        logger.info(f"Content split into {len(self.chunks)} chunks")
        self.cuda_cleanup()
        return self.chunks
    
    def calculate_embeddings(self):
        self.embeddings = []
        batch_size = 16 
        total_chunks = len(self.chunks)
        
        logger.info(f"Starting batch embedding calculation for {total_chunks} chunks...")
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = self.chunks[i:i+batch_size]
    
            inputs = self.tokenizer(
                batch_chunks,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)

                for embedding in batch_embeddings:
                    self.embeddings.append(embedding.cpu().numpy())
            
            processed = min(i + batch_size, total_chunks)
            logger.info(f"Calculated embeddings: {processed}/{total_chunks} chunks")
            self.cuda_cleanup()  
        
        logger.info(f"Completed embedding calculation for {total_chunks} chunks")
        return self.embeddings


    def load_model(self , model_name=Config.EMBEDDING_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        return self.tokenizer, self.model
    
    def load_in_chromadb(self, file_name):
        collection = self.db.get_or_create_collection(name=file_name)
        
        batch_size = 50  
        total_chunks = len(self.chunks)
        
        logger.info(f"Loading {total_chunks} chunks into ChromaDB in batches...")
        
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            
            batch_documents = self.chunks[i:batch_end]
            batch_embeddings = [embedding.tolist() for embedding in self.embeddings[i:batch_end]]
            batch_metadatas = [{"chunk_id": j} for j in range(i, batch_end)]
            batch_ids = [f"{file_name}_{j}" for j in range(i, batch_end)]
            
            collection.add(
                documents=batch_documents,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
                ids=batch_ids
            )
            
            logger.info(f"ChromaDB progress: {batch_end}/{total_chunks} chunks loaded")
            self.cuda_cleanup()
        
        logger.info(f"Successfully loaded all {total_chunks} chunks into ChromaDB collection: {file_name}")

    

    def summarize(self, batch_size=2):
        self.chunks = self.chunk_content(chunk_size=8000, chunk_overlap=400)
        if not self.chunks:
            logger.warning("No chunks to summarize.")
            return []

        logger.info("Starting summarization process with batching...")
        log_to_client("Starting summarization process with batching...")

        try:
            model_name = Config.SUMMARIZATION_MODEL
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize summarizer model/tokenizer: {e}")
            return None

        self.summaries = []
        max_input_tokens = 1024
        total_chunks = len(self.chunks)
        self.cuda_cleanup()

        for i in range(0, total_chunks, batch_size):
            batch_chunks = self.chunks[i:i + batch_size]
            inputs = tokenizer(
                batch_chunks,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens
            ).to(self.device)

            with torch.no_grad():
                summary_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=250,
                    min_length=150,
                    do_sample=False
                )

            decoded_summaries = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            logger.info(f"Summarized batch {i + 1}-{i + len(batch_chunks)} of {total_chunks}")
            self.summaries.extend(decoded_summaries)
            self.cuda_cleanup()

        os.makedirs(f"{self.output_path}/summarize", exist_ok=True)
        for i, summary in enumerate(self.summaries):
            with open(f"{self.output_path}/summarize/summarized_chunk_{i + 1}.md", "w", encoding="utf-8") as f:
                f.write(summary)

        return self.summaries


        
    def cuda_cleanup(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()
        


    def unify_summarized_text(self, output_file="unified_summary.md"):
        self.cuda_cleanup()
        summaries = self.summaries
        if not summaries:
            logger.warning("No summaries available to unify.")
            return None
        

        unified_blocks = []
        
        logger.info(f"Processing {len(summaries)} summaries for unification...")
        with open(f"{self.output_path}/{output_file}", "w", encoding="utf-8") as f:
        
            for idx, summary in enumerate(summaries, start=1):
                if not summary.strip():
                    logger.warning(f"Skipping empty summary block {idx}")
                    continue
                
                # Sanitize and wrap each summary block
                sanitized_summary = sanitize_text(summary)
                wrapped_summary = textwrap.fill(sanitized_summary, width=120)
                
                # Write to file
                f.write(f"### Summary Block {idx}\n\n")
                f.write(wrapped_summary + "\n\n")
                
                unified_blocks.append(wrapped_summary)

        logger.info(f"Unified {len(unified_blocks)} summary blocks.")
        return unified_blocks

def sanitize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    #  Convert specific Unicode punctuation to ASCII equivalents
    replacements = {
        '…': '...',   # ellipsis
        '’': "'",     # right single quote
        '‘': "'",     # left single quote
        '“': '"',     # left double quote
        '”': '"',     # right double quote
        '—': '-',     # em dash
        '–': '-',     # en dash
        '‑': '-',     # non-breaking hyphen
        '•': '-',     # bullet point
        '·': '-',     # middle dot
        '\u00A0': ' ',  # non-breaking space
        '\u202F': ' ',  # narrow no-break space
        '\u200B': '',   # zero-width space
        '\u200C': '',   # zero-width non-joiner
        '\u200D': '',   # zero-width joiner
        '\uFEFF': '',   # zero-width no-break space (BOM)
    }

    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    text = re.sub(r'[\u200e\u200f\u202a-\u202e\u2066-\u2069]', '', text)
    text = ''.join(
        c for c in text
        if c in ('\n', '\t') or (ord(c) >= 32 and unicodedata.category(c)[0] != 'C')
    )
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


