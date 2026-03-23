
import streamlit as st
import os
from utils.helpers import load_dotenv, format_confidence, truncate_text

load_dotenv()

st.set_page_config(
    page_title="Smart Document Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def load_ocr():
    """Load OCR components once at startup."""
    from ocr import TextExtractor
    return TextExtractor(languages=["de", "en"])

@st.cache_resource
def load_nlp():
    """Load NLP components once at startup."""
    from nlp import EntityExtractor, DocumentSummarizer
    extractor = EntityExtractor()
    summarizer = DocumentSummarizer(model_key="en")  # Change to "de" for multilingual
    return extractor, summarizer

@st.cache_resource
def load_embedder():
    """Load embedding model once at startup."""
    from llm import DocumentEmbedder
    return DocumentEmbedder()

def init_session_state():
    defaults = {
        "ocr_result": None,
        "nlp_result": None,
        "summary_result": None,
        "chatbot": None,
        "chat_history": [],
        "document_indexed": False,
        "current_file_name": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

with st.sidebar:
    st.title("Smart Document Analyzer")
    st.caption("CV · NLP · LLMs — one pipeline")

    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Get one at console.anthropic.com — needed for the chat feature",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    st.markdown("**About this project**")
    st.markdown("""
    This app demonstrates:
    - **Phase 1:** EasyOCR text extraction
    - **Phase 2:** spaCy NER + HuggingFace summarization  
    - **Phase 3:** LangChain RAG + Claude chat
    
    Try uploading a German receipt or invoice!
    """)

    if st.session_state.current_file_name:
        st.success(f"Loaded: {st.session_state.current_file_name}")

st.header("Smart Document Analyzer")
st.write("Upload a document image to extract text, analyze it, and chat with it.")

st.subheader("Step 1 — Upload Document")

uploaded_file = st.file_uploader(
    "Upload a document image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Works best with clear photos of printed documents (invoices, receipts, letters)"
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(uploaded_file, caption="Uploaded document", use_container_width=True)

    with col2:
        # Only re-run OCR if the file changed
        if uploaded_file.name != st.session_state.current_file_name:
            with st.spinner("Running OCR (extracting text from image)..."):
                try:
                    ocr = load_ocr()
                    image_bytes = uploaded_file.read()
                    result = ocr.extract_from_bytes(image_bytes)

                    st.session_state.ocr_result = result
                    st.session_state.current_file_name = uploaded_file.name
                    st.session_state.document_indexed = False  # Reset chat index
                    st.session_state.chat_history = []

                except Exception as e:
                    st.error(f"OCR failed: {e}")
                    st.info("Make sure EasyOCR is installed: pip install easyocr")

        if st.session_state.ocr_result:
            result = st.session_state.ocr_result

            st.metric("Words extracted", result.word_count)
            st.metric("OCR confidence", format_confidence(result.confidence))
            st.metric("Detected language", result.language_detected.upper())

            if not result.is_reliable():
                st.warning("Low confidence — try a clearer, well-lit image")

if st.session_state.ocr_result and st.session_state.ocr_result.raw_text:
    st.divider()
    st.subheader("Step 2 — Extracted Text & NLP Analysis")

    ocr_result = st.session_state.ocr_result

    edited_text = st.text_area(
        "Extracted text (you can correct OCR errors here)",
        value=ocr_result.raw_text,
        height=200,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Run NLP Analysis", type="primary"):
            with st.spinner("Running NER and summarization..."):
                try:
                    entity_extractor, summarizer = load_nlp()
                    lang = ocr_result.language_detected

                    nlp_result = entity_extractor.extract(edited_text, language=lang)
                    st.session_state.nlp_result = nlp_result

                    summary = summarizer.summarize(
                        edited_text,
                        language=lang,
                        bullet_points=True,
                    )
                    st.session_state.summary_result = summary

                    st.success("NLP analysis complete!")
                except Exception as e:
                    st.error(f"NLP failed: {e}")
                    st.info("Check that spaCy models are installed")

    with col2:
        if st.session_state.nlp_result:
            nlp = st.session_state.nlp_result
            st.metric("Entities found", len(nlp.entities))
            st.metric("Sentences", len(nlp.sentences))

    if st.session_state.nlp_result:
        nlp = st.session_state.nlp_result
        summary = st.session_state.summary_result

        tab1, tab2, tab3 = st.tabs(["Named Entities", "Summary", "Key Phrases"])

        with tab1:
            if nlp.entities_by_type:
                for label, entities in nlp.entities_by_type.items():
                    entity_texts = [e.text for e in entities]
                    entity_label = entities[0].label_description if entities else label
                    st.write(f"**{entity_label}**")
                    st.write(", ".join(entity_texts))
            else:
                st.info("No named entities found. The text may be too short or unclear.")

        with tab2:
            if summary:
                st.write(summary.get("summary", "No summary generated."))
                stats = summary.get("stats", {})
                if stats:
                    cols = st.columns(3)
                    cols[0].metric("Original words", stats.get("original_words", 0))
                    cols[1].metric("Summary words", stats.get("summary_words", 0))
                    cols[2].metric("Compression", f"{stats.get('compression_ratio', 0):.0%}")

        with tab3:
            if nlp.key_phrases:
                for phrase in nlp.key_phrases[:10]:
                    st.write(f"• {phrase}")
            else:
                st.info("No key phrases extracted.")

if st.session_state.ocr_result and st.session_state.ocr_result.raw_text:
    st.divider()
    st.subheader("Step 3 — Chat with Your Document")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.warning("Enter your Anthropic API key in the sidebar to enable chat.")
    else:
        if not st.session_state.document_indexed:
            with st.spinner("Indexing document for semantic search..."):
                try:
                    from llm import DocumentChatbot
                    embedder = load_embedder()

                    text = st.session_state.ocr_result.raw_text
                    embedder.index_document(text)

                    chatbot = DocumentChatbot(embedder)
                    st.session_state.chatbot = chatbot
                    st.session_state.document_indexed = True

                    with st.spinner("Getting document overview..."):
                        overview = chatbot.get_document_overview()
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"**Document overview:**\n\n{overview}"
                        })

                except Exception as e:
                    st.error(f"Failed to initialize chat: {e}")

        for message in st.session_state.chat_history:
            role = message["role"]
            with st.chat_message(role):
                st.markdown(message["content"])

        if user_input := st.chat_input("Ask anything about this document..."):

            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.chat_history.append({
                "role": "user", "content": user_input
            })

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        chatbot = st.session_state.chatbot
                        response = chatbot.chat(user_input)
                        st.markdown(response)
                        st.session_state.chat_history.append({
                            "role": "assistant", "content": response
                        })
                    except Exception as e:
                        error_msg = f"Chat error: {e}"
                        st.error(error_msg)

        if st.session_state.chat_history:
            if st.button("Clear conversation"):
                st.session_state.chat_history = []
                if st.session_state.chatbot:
                    st.session_state.chatbot.reset_conversation()
                st.rerun()

st.divider()
st.caption(
    "Built with EasyOCR · spaCy · Hugging Face · LangChain · Claude · Streamlit"
)
