
import easyocr
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .preprocessor import ImagePreprocessor


@dataclass
class OCRResult:
   
    raw_text: str                          # Full extracted text as one string
    blocks: list = field(default_factory=list)  # Individual text blocks with positions
    language_detected: str = "unknown"
    confidence: float = 0.0               # Average confidence (0.0 to 1.0)
    word_count: int = 0

    def is_reliable(self, threshold: float = 0.5) -> bool:
        """Returns True if OCR confidence is above threshold."""
        return self.confidence >= threshold

class TextExtractor:
   
    DEFAULT_LANGUAGES = ["de", "en"]  # de = Deutsch (German)

    def __init__(
        self,
        languages: Optional[list] = None,
        use_gpu: bool = False,         # Set True if you have an NVIDIA GPU
        confidence_threshold: float = 0.3
    ):
       
        self.languages = languages or self.DEFAULT_LANGUAGES
        self.confidence_threshold = confidence_threshold

        print(f"Loading OCR model for languages: {self.languages}")
        print("(First run downloads ~100MB model — this is normal)")

        self.reader = easyocr.Reader(
            self.languages,
            gpu=use_gpu,
            verbose=False   # Suppress EasyOCR's internal logging
        )

        self.preprocessor = ImagePreprocessor()

    def extract(self, image_input, preprocess: bool = True) -> OCRResult:
        
        if preprocess:
            image = self.preprocessor.process(image_input)
        else:
            image = image_input

        raw_blocks = self.reader.readtext(image)

        # Step 3: Filter and structure the results
        return self._parse_results(raw_blocks)

    def extract_from_bytes(self, image_bytes: bytes) -> OCRResult:
    
        return self.extract(image_bytes)

    def _parse_results(self, raw_blocks: list) -> OCRResult:
        
        filtered_blocks = []
        all_text_parts = []
        confidence_scores = []

        for bbox, text, score in raw_blocks:
            
            if score < self.confidence_threshold:
                continue

            text = text.strip()
            if not text:
                continue

            filtered_blocks.append({
                "text": text,
                "confidence": round(score, 3),
                "bbox": bbox,               # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                "bbox_simple": self._simplify_bbox(bbox)  # (x, y, width, height)
            })

            all_text_parts.append(text)
            confidence_scores.append(score)

        raw_text = self._join_text_blocks(filtered_blocks)

        avg_confidence = (
            float(np.mean(confidence_scores)) if confidence_scores else 0.0
        )

        return OCRResult(
            raw_text=raw_text,
            blocks=filtered_blocks,
            language_detected=self._guess_language(raw_text),
            confidence=avg_confidence,
            word_count=len(raw_text.split())
        )

    def _simplify_bbox(self, bbox: list) -> tuple:
       
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x = int(min(xs))
        y = int(min(ys))
        w = int(max(xs) - min(xs))
        h = int(max(ys) - min(ys))
        return (x, y, w, h)

    def _join_text_blocks(self, blocks: list) -> str:
        
        if not blocks:
            return ""

        sorted_blocks = sorted(blocks, key=lambda b: b["bbox"][0][1])

        result_lines = []
        current_line = []
        prev_y = None
        line_height_threshold = 20  # pixels — tune this if results look wrong

        for block in sorted_blocks:
            y = block["bbox"][0][1]  # Top-left y coordinate

            if prev_y is None or abs(y - prev_y) < line_height_threshold:
                current_line.append(block["text"])  # Same line
            else:
                result_lines.append(" ".join(current_line))  # New line
                current_line = [block["text"]]

            prev_y = y

        if current_line:
            result_lines.append(" ".join(current_line))

        return "\n".join(result_lines)

    def _guess_language(self, text: str) -> str:
        
        german_chars = set("äöüÄÖÜß")
        german_words = {"der", "die", "das", "und", "ist", "nicht", "mit", "für"}

        text_lower = text.lower()
        words = set(text_lower.split())

        has_german_chars = any(c in text for c in german_chars)
        has_german_words = bool(words & german_words)

        if has_german_chars or has_german_words:
            return "de"
        return "en"

if __name__ == "__main__":
    import sys
    import cv2

    print("=== TextExtractor Test ===\n")

    img = np.ones((300, 700, 3), dtype=np.uint8) * 255
    cv2.putText(img, "Rechnung Nr. 2024-001", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Betrag: 149,99 EUR", (30, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Datum: 22. März 2026", (30, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    image_path = sys.argv[1] if len(sys.argv) > 1 else img

    extractor = TextExtractor()
    result = extractor.extract(image_path)

    print(f"Extracted text:\n{'-'*40}")
    print(result.raw_text)
    print(f"\nStats:")
    print(f"  Words:      {result.word_count}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Language:   {result.language_detected}")
    print(f"  Reliable:   {result.is_reliable()}")
    print(f"  Blocks:     {len(result.blocks)}")
