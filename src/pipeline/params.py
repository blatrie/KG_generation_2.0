"""
This script sets up and configures multiple NLP models and tools to process text data, including sentence embeddings, sequence-to-sequence models, and semantic segmentation. 
The setup involves using both the Hugging Face `transformers` library for pre-trained sequence models and the `sentence-transformers` library for generating sentence embeddings. 
Additionally, the `spaCy` library is used for tokenization and natural language processing (NLP) tasks.

Key Functionalities:

1. **Device Setup for Computation (CUDA/MPS/CPU)**:
    - The script automatically selects the appropriate computation device for model execution (GPU, MPS, or CPU) based on the available hardware. It checks for CUDA (for NVIDIA GPUs), MPS (for Apple M1/M2 chips), or defaults to the CPU if no compatible device is found.

2. **Model and Tokenizer Initialization**:
    - **Sentence Transformer (`merge_model`)**:
        - The `sentence-transformers/all-MiniLM-L6-v2` model is loaded to generate high-quality sentence embeddings. This model is used for tasks like text similarity, clustering, or any task requiring semantic representation of sentences.
    - **Translation Model (`rdf_model`)**:
        - The `Babelscape/mrebel-large` model is loaded for sequence-to-sequence tasks like machine translation. The model and tokenizer are prepared to handle text in English or other languages (with `src_lang` and `tgt_lang` parameters).
    - **Tokenizer for Semantic Segmentation (`tokenizer_semantic_seg`)**:
        - The `sentence-transformers/all-MiniLM-L6-v2` tokenizer is also loaded here, likely used for preparing data for semantic segmentation or sentence-level classification tasks.

3. **spaCy Setup**:
    - The script downloads and loads the `en_core_web_sm` spaCy model, which is used for tokenization, part-of-speech tagging, and other NLP preprocessing tasks on English text.
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer
import spacy

spacy.cli.download("en_core_web_sm")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device : {DEVICE}")

merge_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained("Babelscape/mrebel-large", tgt_lang="en_XX")  # you can add src_lang= "en_XX" or "fr_XX to add an input language. Makes the model more precise !
rdf_model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/mrebel-large").to(DEVICE)
tokenizer_semantic_seg = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

nlp = spacy.load("en_core_web_sm")  # For english
