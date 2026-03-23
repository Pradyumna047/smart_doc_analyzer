
from torch.mtia import device
from transformers import pipeline, Pipeline
from typing import Optional
import re

class DocumentSummarizer:
   
    MODELS = {
        "en": "sshleifer/distilbart-cnn-12-6",       # Fast, good English quality
        "de": "csebuetnlp/mT5_multilingual_XLSum",   # Supports German & English
        "multilingual": "csebuetnlp/mT5_multilingual_XLSum",
    }

    LANG_TOKENS = {
        "de": "german",
        "en": "english",
    }

    def __init__(self, model_key: str = "en", device: int = -1):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        model_name = self.MODELS.get(model_key, self.MODELS["en"])
        self.model_key = model_key

        print(f"Loading summarization model: {model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._pipeline = None  # not used
        print("Summarization model ready.")

    def summarize(
        self,
        text: str,
        language: str = "en",
        max_length: int = 150,   # Max tokens in output summary
        min_length: int = 40,    # Min tokens (prevent too-short summaries)
        bullet_points: bool = False,
    ) -> dict:
        if not text.strip():
            return {"summary": "", "stats": {}}

        chunks = self._chunk_text(text, max_chunk_tokens=400)

        chunk_summaries = []
        for chunk in chunks:
            prepared = self._prepare_input(chunk, language)
            summary = self._summarize_chunk(
                prepared,
                max_length=max_length // len(chunks) + 20,
                min_length=min(min_length, 20),
            )
            chunk_summaries.append(summary)

        if len(chunk_summaries) > 1:
            combined = " ".join(chunk_summaries)
            final_summary = self._summarize_chunk(
                combined, max_length=max_length, min_length=min_length
            )
        else:
            final_summary = chunk_summaries[0]

        result = {
            "summary": final_summary,
            "stats": {
                "original_words": len(text.split()),
                "summary_words": len(final_summary.split()),
                "compression_ratio": round(
                    len(final_summary.split()) / max(len(text.split()), 1), 2
                ),
                "chunks_processed": len(chunks),
                "language": language,
            }
        }

        if bullet_points:
            result["bullet_points"] = self._extract_bullets(text, final_summary)

        return result

    def quick_summary(self, text: str, language: str = "en") -> str:
        """Returns just the summary string — simpler interface."""
        return self.summarize(text, language=language)["summary"]

    def _chunk_text(self, text: str, max_chunk_tokens: int = 400) -> list:
        max_words = int(max_chunk_tokens * 0.75)
        words = text.split()

        if len(words) <= max_words:
            return [text]  # Short enough, no chunking needed

        chunks = []
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence.split())
            if current_len + sentence_len > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_len = sentence_len
            else:
                current_chunk.append(sentence)
                current_len += sentence_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _prepare_input(self, text: str, language: str) -> str:
        
        if self.model_key == "de":
            lang_token = self.LANG_TOKENS.get(language, "english")
            return f"{lang_token}: {text}"
        return text

    def _summarize_chunk(self, text: str, max_length: int, min_length: int) -> str:
        try:
            import torch
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            summary_ids = self._model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )
            return self._tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
            ).strip()
        except Exception as e:
            print(f"Summarization error: {e}")
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return " ".join(sentences[:3])

    def _extract_bullets(self, original: str, summary: str) -> list:
       
        sentences = re.split(r'(?<=[.!?])\s+', original)

        summary_words = set(summary.lower().split())

        def score(sent):
            words = sent.lower().split()
            if len(words) < 5 or len(words) > 40:
                return 0
            overlap = len(set(words) & summary_words)
            return overlap

        scored = [(score(s), s) for s in sentences]
        scored.sort(reverse=True)

        bullets = [s for _, s in scored[:5] if _]
        return bullets

if __name__ == "__main__":
    sample_text = """
    Anthropic has developed Claude, an AI assistant designed to be helpful, 
    harmless, and honest. The company, founded in 2021 by former OpenAI researchers,
    focuses on AI safety research and building reliable AI systems. Claude is used 
    by businesses and individuals for tasks ranging from writing and analysis to 
    coding and research. The latest models support long context windows of up to 
    200,000 tokens, enabling processing of entire books or codebases. Anthropic 
    has raised significant funding to continue its research into making AI systems 
    safer and more interpretable. The company publishes research on topics like 
    constitutional AI and mechanistic interpretability to advance the broader field.
    """

    print("=== DocumentSummarizer Test ===\n")
    summarizer = DocumentSummarizer(model_key="en")

    result = summarizer.summarize(sample_text, bullet_points=True)

    print(f"Summary:\n{result['summary']}\n")
    print(f"Stats: {result['stats']}")
    if result.get("bullet_points"):
        print(f"\nKey points:")
        for bullet in result["bullet_points"]:
            print(f"  • {bullet}")
