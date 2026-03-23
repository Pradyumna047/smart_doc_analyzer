
import os
import anthropic
from typing import Optional
from .embedder import DocumentEmbedder


class DocumentChatbot:
    
    SYSTEM_PROMPT = """You are a helpful document analysis assistant.
    
Your job is to answer questions about a document that has been provided to you.

RULES:
1. Only answer based on the document context provided in each message.
2. If the answer is not in the provided context, say: "I couldn't find that information in the document."
3. Be concise and specific. Quote relevant parts of the document when helpful.
4. If the document appears to be in German, you may answer in German or English based on the user's question language.
5. For numbers, amounts, and dates — be exact. Don't approximate.

You are not a general assistant. Focus exclusively on the document's content."""

    def __init__(
        self,
        embedder: DocumentEmbedder,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_context_chunks: int = 3,
    ):
        
        self.embedder = embedder
        self.max_context_chunks = max_context_chunks
        self.model = model

        self.client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.history: list = []

    def chat(self, user_message: str) -> str:
        context = self.embedder.get_context_for_query(
            user_message,
            top_k=self.max_context_chunks
        )
        augmented_message = self._build_augmented_message(user_message, context)

        self.history.append({"role": "user", "content": augmented_message})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=self.SYSTEM_PROMPT,
            messages=self.history,   # Send full conversation history
        )

        assistant_message = response.content[0].text

        self.history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message

    def reset_conversation(self):
        """Clear chat history — start a fresh conversation."""
        self.history = []

    def get_document_overview(self) -> str:
        
        return self.chat(
            "Please give me a brief overview of what this document is about. "
            "What type of document is it, and what are the key pieces of information it contains?"
        )

    def _build_augmented_message(self, user_message: str, context: str) -> str:
        
        if not context:
            # No relevant chunks found — tell Claude explicitly
            return f"""<document_context>
No relevant context found in the document for this question.
</document_context>

<question>
{user_message}
</question>"""

        return f"""<document_context>
The following excerpts from the document are most relevant to the question:

{context}
</document_context>

<question>
{user_message}
</question>"""

    def _trim_history_if_needed(self, max_turns: int = 20):
        
        if len(self.history) > max_turns * 2:
            # Keep first message (document overview) + last N-1 turns
            self.history = self.history[:2] + self.history[-(max_turns * 2 - 2):]

if __name__ == "__main__":
    import sys
    from .embedder import DocumentEmbedder

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set your API key first:")
        print("  export ANTHROPIC_API_KEY='your-key-here'")
        sys.exit(1)

    sample_doc = """
    Rechnung Nr. 2024-0042
    Datum: 15. März 2024
    TechShop Berlin GmbH — Alexanderplatz 1, 10178 Berlin
    
    Für: Anna Schmidt, Unter den Linden 5, 10117 Berlin
    
    Position 1: Dell XPS 15 Laptop - 1.299,00 EUR
    Position 2: USB-C Hub 7-Port - 49,99 EUR  
    Position 3: Laptop-Rucksack - 89,00 EUR
    
    Zwischensumme: 1.437,99 EUR
    MwSt. (19%): 273,22 EUR
    Gesamtbetrag: 1.711,21 EUR
    
    Zahlungsziel: 30 Tage nach Rechnungsdatum
    """

    print("=== DocumentChatbot Test ===\n")
    embedder = DocumentEmbedder()
    embedder.index_document(sample_doc, chunk_size=60, overlap=15)

    chatbot = DocumentChatbot(embedder)

    questions = [
        "What is the total amount to pay?",
        "Who is the seller?",
        "What items were purchased?",
        "What is the payment deadline?",
    ]

    for question in questions:
        print(f"User: {question}")
        answer = chatbot.chat(question)
        print(f"Claude: {answer}\n")
