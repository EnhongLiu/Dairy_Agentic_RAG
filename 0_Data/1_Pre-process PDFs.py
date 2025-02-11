## Read in PDF example. If this has been done, just skip to read local chroma_db

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)  
logger.info("Starting document processing.")

# Define the source directory and initialize variables
source_dir = 'xxxxx'
converter = DocumentConverter()
docs = []

# Convert each PDF in the directory to a document
for pdf_path in Path(source_dir).glob("*.pdf"):
    logger.info(f"Processing file: {pdf_path}")
    docs.append(converter.convert(str(pdf_path)).document)

# Ensure documents were found
if not docs:
    logger.warning("No documents found for processing.")
else:
    logger.info(f"Successfully processed {len(docs)} documents.")

# Split text into chunks
chunker = HybridChunker()  # Optionally specify a tokenizer from transformers- huggingface
chunk_iter = []

# Process each document individually to avoid AttributeError
for doc in docs:
    try:
        chunks = chunker.chunk(doc)
        chunk_iter.extend(chunks)  # Collect chunks from each document
    except AttributeError as e:
        logger.error(f"Error processing document: {e}")
