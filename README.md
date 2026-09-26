# OffOn Restaurant AI Assistant

The OffOn restaurant AI assistant is a **RAG-based** question-answering system. It reads restaurant and menu information from a JSON file, creates embeddings for the documents, classifies user questions, and generates answers based on the restaurant's actual information.

The following tools were used to build this project:

- Python
- FastAPI
- LangGraph
- OpenAI-compatible API
- Sentence Transformers
- Hybrid Search
- SQLite Conversation Memory
- Telegram Bot with `pyTelegramBotAPI`

## Features

- Answers questions about the menu, ingredients, prices, and availability
- Answers questions about the address, phone number, opening hours, and general restaurant information
- Detects general food-related questions
- Combines semantic and keyword search
- Handles Persian spelling variations in menu item names
- Performs full catalog retrieval for list and comparison questions
- Provides a dedicated food and calorie analysis mode
- Stores conversation memory with SQLite
- Limits the number of stored messages per session
- Supports streaming responses through FastAPI and Telegram
- Caches the embedding model and RAG resources
- Retries failed model requests and handles errors
- Logs request duration, response duration, request IDs, and token usage
- Prevents the model from executing raw SQL

## Overall Architecture

The main question-answering flow is:

```text
User Question
      |
      v
Conversation Memory
      |
      v
LLM Router
      |
      +--> restaurant_specific
      |        |
      |        v
      |   Hybrid Search or Full Catalog Search
      |        |
      |        v
      |   Grounded Answer Generation
      |
      +--> general_food
      |        |
      |        v
      |   General Food Answer
      |
      +--> ambiguous / out_of_scope
               |
               v
          Controlled Response
```

### Restaurant Question Flow

1. The Router converts the question into a `QuestionRoute`.
2. The application identifies the operation: single-item question, list, comparison, filter, or aggregate.
3. Normal questions use Hybrid Search with `TOP_K=3`.
4. List, comparison, filter, and aggregate questions inspect the complete relevant catalog.
5. Address and contact questions directly read the dedicated `restaurant_information` document.
6. The fixed restaurant profile is sent to the final model together with the relevant documents.
7. The answer is generated only from valid context.

## Project Structure

```text
.
├── app/
│   ├── agent/
│   │   ├── agent_state.py       # Shared LangGraph state
│   │   ├── router.py            # Question classification and query-plan creation
│   │   ├── workflow.py          # LangGraph construction and execution
│   │   └── workflow_nodes.py    # Route, retrieval, and generation nodes
│   ├── core/
│   │   ├── embedding_model.py   # Cached embedding model loader
│   │   ├── llm_client.py        # LLM client, retries, and logging
│   │   └── request_context.py   # Request IDs used by logs
│   ├── memory/
│   │   └── conversation_memory.py # Short-term SQLite memory
│   ├── rag/
│   │   ├── rag_qa.py            # RAG prompts and answer generation
│   │   ├── retrieve.py          # Normalization and Hybrid Search
│   │   └── retrieval_resources.py # Cached model, documents, and embeddings
│   ├── telegram/
│   │   ├── bot.py               # Telegram bot entry point
│   │   ├── handlers.py          # Message and command handlers
│   │   └── sender.py            # Streaming response rendering in Telegram
│   └── api.py                   # Main FastAPI application
├── data/
│   ├── offonburg_clean_rag.json       # Main information source
│   ├── offonburg_documents.json       # Documents prepared for embedding
│   ├── offonburg_embedded_documents.json # Documents with embedding indexes
│   └── offonburg_embeddings.npy       # Embedding vectors
├── scripts/
│   ├── rag_loader.py            # Converts source JSON into documents
│   ├── embed_documents.py       # Creates offline embeddings
│   └── measure_api.py           # Measures client and server timing
├── requirements.txt
├── .gitignore
└── README.md
```

## Prerequisites

- Python 3.10 or newer; Python 3.12 is recommended.
- An API key from an OpenAI Chat Completions-compatible provider
- A Telegram bot token if the Telegram bot is used
- The downloaded embedding model, or temporary access to download it

## Installation

Run the following commands from the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```powershell
.venv\Scripts\activate
```

## Environment Configuration

Create a `.env` file in the project root. This file must not be committed.

```env
API_KEY=your_provider_api_key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL_NAME=your-model-name
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_STREAM_EDIT_INTERVAL_SECONDS=0.8
MEMORY_MAX_MESSAGES=10
EMBEDDING_LOCAL_FILES_ONLY=true
```

Important notes:

- `LLM_MODEL_NAME` must be the exact model name supported by the provider.
- `MEMORY_MAX_MESSAGES` is the number of messages, not the number of turns; use an even value.
- When `EMBEDDING_LOCAL_FILES_ONLY=true`, the application does not connect to the internet to load the embedding model at runtime.

## Preparing RAG Data

The main source file is:

```text
data/offonburg_clean_rag.json
```

### Restaurant Information

General restaurant information belongs in the `restaurant` section. The `profile_text` field stores free-form restaurant information:

```json
{
  "restaurant": {
    "name": "مجموعه غذایی آفان",
    "address": "آدرس رستوران",
    "phone": "شماره تلفن",
    "profile_text": "متن معرفی رستوران، معنی نام، سال تأسیس، مشخصات مدیر و داستان رستوران.",
    "food_types": ["فست‌فود", "ساندویچ"],
    "service_hours": {
      "days": "شنبه تا جمعه",
      "raw": "از ساعت ۱۲ تا ۲۳:۵۹"
    }
  }
}
```

### Menu Information

Each menu item belongs in the `menu` array:

```json
{
  "name": "آن پرایم",
  "category": "ساندویچ",
  "ingredients": ["برگر ۱۴۰ گرمی", "گوجه", "نان"],
  "price_toman": 640000,
  "available": "available",
  "source_description": "توضیحات غذا"
}
```

After every change to the JSON source, rebuild the documents and embeddings:

```bash
./.venv/bin/python -m scripts.rag_loader
./.venv/bin/python -m scripts.embed_documents
```

Avoid running `scripts/embed_documents.py` directly. Running it as a module with `-m` ensures that project imports are resolved correctly.

## Running the API

Run from the project root:

```bash
./.venv/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000
```

For development with automatic reload:

```bash
./.venv/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

The embedding model and RAG documents are loaded during startup and cached within the same process.

## API Endpoints

### Health Check

```http
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

### Complete Response

```http
POST /ask
Content-Type: application/json
```

Request body:

```json
{
  "session_id": "user-123",
  "question": "همبرگرهای موجود را نام ببر"
}
```

Example response:

```json
{
  "question": "همبرگرهای موجود را نام ببر",
  "route": "restaurant_specific",
  "answer": "...",
  "sources": [
    {
      "item_name": "آن پرایم",
      "document_type": "menu_item",
      "hybrid_score": 0.91
    }
  ]
}
```

### Streaming Response with Server-Sent Events

```http
POST /ask/stream
Content-Type: application/json
```

Test with curl:

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"stream-test","question":"آدرس رستوران کجاست؟"}'
```

Streaming events include:

- `start`: workflow started
- `route`: selected route and query plan
- `sources`: metadata for retrieved documents
- `token`: a generated response fragment
- `done`: response completed
- `error`: controlled error

## Running the Telegram Bot

After adding `TELEGRAM_BOT_TOKEN` to `.env`, run:

```bash
./.venv/bin/python -m app.telegram.bot
```

Bot features:

- Uses `pyTelegramBotAPI`
- Automatically creates a `session_id` from chat and user identifiers
- Separates user memory in group chats
- Displays responses progressively by editing a Telegram message
- Splits responses longer than Telegram's message limit
- Uses the same workflow as the FastAPI service

Available commands:

- `/start`
- `/help`
- Regular text messages

## Conversation Memory

Conversation memory is stored in the following SQLite file:

```text
memory.db
```

User and assistant messages are stored for each `session_id`. The default value of `MEMORY_MAX_MESSAGES=10` keeps five complete turns. Older messages are removed automatically.

The API client must provide `session_id`. For Telegram, the session ID is generated automatically.

The `memory.db` file is runtime data and must not be committed to GitHub.

## Hybrid Search

Normal retrieval combines two signals:

```text
hybrid_score = 0.7 * semantic_score + 0.3 * keyword_score
```

- The semantic score is calculated with `paraphrase-multilingual-MiniLM-L12-v2`.
- The keyword score uses token overlap and fuzzy matching for menu item names.
- Persian and Arabic character variations such as `ي/ی`, `ك/ک`, and `آ/ا` are normalized.
- The fuzzy menu-name threshold is `0.82`.
- Normal questions return only the top three documents.
- List, comparison, filter, and aggregate questions use the complete relevant catalog.

## Router and Query Plan

The Router produces the following structured fields:

- `category`: restaurant-specific, general food, out of scope, or ambiguous
- `operation`: single item, list, comparison, filter, or aggregate
- `domain`: menu, ingredients, nutrition, price, availability, or restaurant information
- `menu_category`: detected menu category when available
- `target_item`: canonical menu item name when detected
- `requires_full_catalog`: whether the complete catalog must be inspected
- `calorie_mode`: not requested, exact only, or estimate if needed

The Router response is validated with Pydantic before it is used. Invalid Router output is retried three times with a short delay.

## Error Handling and Logging

`app/core/llm_client.py` retries timeout, connection, and rate-limit errors with exponential backoff. The application also logs:

- LLM request start and completion
- Provider response duration
- Request ID
- Prompt and completion token counts
- Embedding model loading duration
- Retrieval duration
- Memory loading and saving duration
- Time to first token during streaming

The API also returns these headers:

- `X-Request-ID`
- `X-Server-Time-Ms`

## Measuring API Timing

Start the API first, then run this command in another terminal:

```bash
./.venv/bin/python -m scripts.measure_api "آدرس رستوران کجاست؟"
```

Available options:

```bash
./.venv/bin/python -m scripts.measure_api \
  --url http://127.0.0.1:8000/ask \
  --session-id timing-test \
  "قیمت آن پرایم چقدر است؟"
```

This utility compares client round-trip time with the time measured by the server.

## Running Internal CLI Tools

Run the workflow from the command line:

```bash
./.venv/bin/python -m app.agent.workflow
```

Test retrieval:

```bash
./.venv/bin/python -m app.rag.retrieve
```

Test RAG question answering:

```bash
./.venv/bin/python -m app.rag.rag_qa
```

## Basic Code Verification

To check the syntax of project files:

```bash
./.venv/bin/python -m compileall -q app scripts
```

To verify the generated RAG data, the following must remain aligned:

- The number of documents in `offonburg_embedded_documents.json`
- The number of rows in `offonburg_embeddings.npy`
- The order of documents and embeddings

After changing menu text or source JSON, always run both `rag_loader` and `embed_documents`.

## Current Limitations

- The system currently does not place orders or process online payments.
- Order-related Tool Calling is not included in the current version.
- The source data does not contain complete calorie information; exact answers are provided only when calorie data is available.
- Menu updates are not automatic; the source JSON and embeddings must be rebuilt manually.

## Recommended Next Steps

1. Add automated tests for the Router and retrieval components.
2. Add RAG quality evaluation with a reference question-answer dataset.
3. Add structured logging and production monitoring.

