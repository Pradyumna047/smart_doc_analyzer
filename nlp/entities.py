
import spacy
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


@dataclass
class Entity:
    text: str           # The actual text: "Max Mustermann"
    label: str          # Entity type: "PERSON", "DATE", "MONEY", etc.
    start: int          # Character start position in original text
    end: int            # Character end position
    confidence: float = 1.0  # spaCy NER confidence (if available)

    @property
    def label_description(self) -> str:
        descriptions = {
            "PER": "Person",          "PERSON": "Person",
            "ORG": "Organization",    "LOC": "Location",
            "GPE": "Place",           "DATE": "Date",
            "TIME": "Time",           "MONEY": "Money",
            "CARDINAL": "Number",     "ORDINAL": "Ordinal number",
            "PERCENT": "Percentage",  "PRODUCT": "Product",
            "EVENT": "Event",         "LAW": "Law/Regulation",
            "MISC": "Miscellaneous",
        }
        return descriptions.get(self.label, self.label)


@dataclass
class NLPResult:
    entities: list = field(default_factory=list)    # List[Entity]
    entities_by_type: dict = field(default_factory=dict)  # grouped by label
    sentences: list = field(default_factory=list)   # Sentence boundaries
    tokens: list = field(default_factory=list)      # All tokens (words)
    language: str = "unknown"
    key_phrases: list = field(default_factory=list) # Important noun phrases

    def get_entities_of_type(self, label: str) -> list:
        
        return self.entities_by_type.get(label, [])

    def to_dict(self) -> dict:
        
        return {
            "language": self.language,
            "entity_count": len(self.entities),
            "entities": {
                label: [e.text for e in entities]
                for label, entities in self.entities_by_type.items()
            },
            "sentences": self.sentences,
            "key_phrases": self.key_phrases[:10],  # Top 10
        }


class EntityExtractor:
    
    MODELS = {
        "en": "en_core_web_sm",    # English: small model, fast
        "de": "de_core_news_sm",   # German: newspaper corpus model
    }

    def __init__(self):
        
        self._models = {}
        self._load_models()

    def _load_models(self):

        for lang, model_name in self.MODELS.items():
            try:
                self._models[lang] = spacy.load(model_name)
                print(f"Loaded spaCy model: {model_name}")
            except OSError:
                print(f"WARNING: spaCy model '{model_name}' not found.")
                print(f"  Run: python -m spacy download {model_name}")

    def extract(self, text: str, language: str = "auto") -> NLPResult:
        
        if not text.strip():
            return NLPResult()

        if language == "auto":
            language = self._detect_language(text)

        nlp = self._get_model(language)
        if nlp is None:
            return NLPResult(language=language)

        doc = nlp(text)

        entities = self._extract_entities(doc)
        entities_by_type = self._group_entities(entities)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        key_phrases = self._extract_key_phrases(doc)
        tokens = [token.text for token in doc if not token.is_space]

        return NLPResult(
            entities=entities,
            entities_by_type=entities_by_type,
            sentences=sentences,
            tokens=tokens,
            language=language,
            key_phrases=key_phrases,
        )

    def _extract_entities(self, doc) -> list:
       
        entities = []
        seen = set()  # Deduplicate (same entity appearing multiple times)

        for ent in doc.ents:
            
            key = (ent.text.strip().lower(), ent.label_)
            if key in seen:
                continue
            seen.add(key)

            entities.append(Entity(
                text=ent.text.strip(),
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
            ))

        return entities

    def _group_entities(self, entities: list) -> dict:
        
        grouped = defaultdict(list)
        for entity in entities:
            grouped[entity.label].append(entity)
        return dict(grouped)

    def _extract_key_phrases(self, doc) -> list:
        
        phrases = []
        seen = set()

        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            
            if (len(text) > 3
                    and text.lower() not in seen
                    and not chunk.root.is_stop):
                phrases.append(text)
                seen.add(text.lower())

        return sorted(phrases, key=len, reverse=True)[:20]

    def _detect_language(self, text: str) -> str:
        """Simple German vs English detection."""
        german_indicators = set("äöüÄÖÜß")
        german_words = {
            "der", "die", "das", "und", "ist", "nicht", "mit", "für",
            "auf", "dem", "ein", "eine", "bei", "von", "im", "zu"
        }
        words = set(text.lower().split())
        if any(c in text for c in german_indicators) or words & german_words:
            return "de"
        return "en"

    def _get_model(self, language: str):
        """Get loaded model for a language, with fallback to English."""
        if language in self._models:
            return self._models[language]
        if "en" in self._models:
            print(f"No model for '{language}', falling back to English")
            return self._models["en"]
        print("ERROR: No spaCy models loaded. Run the download commands above.")
        return None
if __name__ == "__main__":
    extractor = EntityExtractor()

    sample_text = """
    Rechnung
    Lieferant: Mustermann GmbH, Berlin
    Rechnungsdatum: 15. März 2024
    Fälligkeitsdatum: 14. April 2024

    Artikel: Laptop Dell XPS 15
    Menge: 2 Stück
    Einzelpreis: 1.299,00 EUR
    Gesamtbetrag: 2.598,00 EUR
    MwSt. (19%): 493,62 EUR

    Bitte überweisen Sie den Betrag auf unser Konto bei der Deutschen Bank.
    IBAN: DE89 3704 0044 0532 0130 00
    """

    print("=== EntityExtractor Test ===\n")
    result = extractor.extract(sample_text)

    print(f"Language: {result.language}")
    print(f"Sentences: {len(result.sentences)}")
    print(f"\nEntities found ({len(result.entities)} total):")
    for label, entities in result.entities_by_type.items():
        entity_texts = [e.text for e in entities]
        print(f"  {label:12} → {entity_texts}")

    print(f"\nKey phrases (top 5):")
    for phrase in result.key_phrases[:5]:
        print(f"  • {phrase}")

    print(f"\nStructured output:")
    import json
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
