# Smart Document Analyzer
### Intelligentes Dokumenten-Analyse-System

A beginner-friendly ML project combining **Computer Vision (OCR)**, **NLP**, and **LLMs** into one pipeline.  
Upload any document image → extract text → summarize → chat with it.

---

## What you'll learn building this
- **Phase 1 (CV/OCR):** OpenCV image preprocessing + EasyOCR text extraction
- **Phase 2 (NLP):** spaCy NER + Hugging Face summarization pipelines
- **Phase 3 (LLMs):** LangChain RAG + Anthropic/OpenAI API chat
- **Phase 4 (Deploy):** Streamlit UI + Hugging Face Spaces deployment

## Tech stack
| Area | Library | Why |
|------|---------|-----|
| Image processing | `opencv-python`, `Pillow` | Industry standard |
| OCR | `easyocr` | Best multilingual support (German works great) |
| NLP | `spacy`, `transformers` | Used at every German AI company |
| LLM | `langchain`, `anthropic` | Real-world RAG pipeline |
| Vector store | `faiss-cpu` | Fast similarity search |
| UI | `streamlit` | Deploy in minutes |

## Project structure
```
smart_doc_analyzer/
├── app.py                  # Streamlit UI (run this)
├── ocr/
│   ├── preprocessor.py     # Image cleaning (Phase 1)
│   └── extractor.py        # EasyOCR text extraction (Phase 1)
├── nlp/
│   ├── entities.py         # spaCy NER (Phase 2)
│   └── summarizer.py       # Hugging Face summarization (Phase 2)
├── llm/
│   ├── embedder.py         # Sentence embeddings + FAISS (Phase 3)
│   └── chat.py             # RAG chat pipeline (Phase 3)
├── utils/
│   └── helpers.py          # Shared utilities
├── tests/
│   └── test_pipeline.py    # Basic tests
├── sample_docs/            # Put test images here
├── requirements.txt
└── README.md
```

## Quickstart
```bash
# 1. Clone and set up environment
git clone <your-repo>
cd smart_doc_analyzer
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download de_core_news_sm   # German model
python -m spacy download en_core_web_sm    # English model

# 3. Add your API key
export ANTHROPIC_API_KEY="your-key-here"

# 4. Run the app
streamlit run app.py
```

## Learning tips
- Read each file top-to-bottom before running it 
- Break something intentionally and fix it — best way to understand OCR preprocessing
- Try uploading a German receipt first — the multilingual support will impress you

---

