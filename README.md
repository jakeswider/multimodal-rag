# Multimodal RAG

Local multimodal RAG for PDFs. Text and images are embedded together with CLIP and stored in pgvector, so a question can pull back a chart just as easily as a paragraph. Answers come from Qwen2.5-VL in Ollama, behind a small Streamlit app.

## Setup

```bash
uv sync
brew install poppler tesseract ollama
ollama pull qwen2.5vl
cp .env.example .env
docker compose up -d db
```

## Usage

Put PDFs in `data/pdfs`, then ingest them:

```bash
uv run python -m ingest.pipeline data/pdfs
```

Start the app:

```bash
uv run streamlit run app.py
```

## Tests

```bash
uv run pytest
```
