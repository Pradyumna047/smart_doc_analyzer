
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_preprocessor_handles_numpy_array():
    """Preprocessor should accept numpy arrays and return grayscale."""
    from ocr.preprocessor import ImagePreprocessor

    preprocessor = ImagePreprocessor()

    test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    test_image[50:150, 50:350] = 0  # Black rectangle (like text)

    result = preprocessor.process(test_image)

    assert result is not None, "Preprocessor returned None"
    assert len(result.shape) == 2, f"Expected grayscale (2D), got shape {result.shape}"
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    print("PASS: test_preprocessor_handles_numpy_array")


def test_preprocessor_handles_upscaling():
    """Small images should be upscaled to at least 1000px wide."""
    from ocr.preprocessor import ImagePreprocessor

    preprocessor = ImagePreprocessor()
    small_image = np.ones((50, 100, 3), dtype=np.uint8) * 255  # Very small

    result = preprocessor.process(small_image)

    assert result.shape[1] >= 1000, f"Image not upscaled, width={result.shape[1]}"
    print("PASS: test_preprocessor_handles_upscaling")

def test_ocr_result_reliability():
    """OCRResult.is_reliable() should correctly threshold confidence."""
    from ocr.extractor import OCRResult

    high_confidence = OCRResult(raw_text="Test", confidence=0.9, word_count=1)
    low_confidence = OCRResult(raw_text="???", confidence=0.2, word_count=1)

    assert high_confidence.is_reliable(), "High confidence should be reliable"
    assert not low_confidence.is_reliable(), "Low confidence should not be reliable"
    print("PASS: test_ocr_result_reliability")

def test_german_language_detection():
    """German text with umlauts should be detected as German."""
    from nlp.entities import EntityExtractor

    extractor = EntityExtractor.__new__(EntityExtractor)  # Skip __init__ (no model load)
    extractor._models = {}

    german_text = "Die Rechnung beträgt 149,99 EUR für den Artikel."
    english_text = "The invoice amounts to 149.99 EUR for the item."

    assert extractor._detect_language(german_text) == "de", "German text not detected"
    assert extractor._detect_language(english_text) == "en", "English text not detected"
    print("PASS: test_german_language_detection")

def test_text_chunking():
    """Chunker should split long text and respect overlap."""
    from llm.embedder import DocumentEmbedder

    embedder = DocumentEmbedder.__new__(DocumentEmbedder)  # Skip model loading

    short_text = "word " * 50  # 50 words
    long_text = "word " * 500  # 500 words

    chunks_short = embedder._create_chunks(short_text, chunk_size=200, overlap=50)
    assert len(chunks_short) == 1, f"Short text should be 1 chunk, got {len(chunks_short)}"

    chunks_long = embedder._create_chunks(long_text, chunk_size=200, overlap=50)
    assert len(chunks_long) > 1, f"Long text should have multiple chunks, got {len(chunks_long)}"

    print(f"PASS: test_text_chunking ({len(chunks_long)} chunks for 500-word text)")

def test_helpers():
    """Test utility functions."""
    from utils.helpers import truncate_text, format_confidence

    long = "a" * 300
    assert len(truncate_text(long, max_chars=100)) == 100
    assert truncate_text("short text") == "short text"

    assert "High" in format_confidence(0.9)
    assert "Medium" in format_confidence(0.6)
    assert "Low" in format_confidence(0.2)

    print("PASS: test_helpers")

if __name__ == "__main__":
    print("Running pipeline tests...\n")

    tests = [
        test_preprocessor_handles_numpy_array,
        test_preprocessor_handles_upscaling,
        test_ocr_result_reliability,
        test_german_language_detection,
        test_text_chunking,
        test_helpers,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}")
            print(f"  Error: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests")

    if failed == 0:
        print("\nAll tests passed! Your pipeline is wired up correctly.")
    else:
        print("\nSome tests failed. Check the errors above.")
