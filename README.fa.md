# OffOn Restaurant AI Assistant

دستیار هوش مصنوعی رستوران آفان یک سامانه پرسش‌وپاسخ مبتنی بر **RAG** است. این پروژه اطلاعات رستوران و منوی غذاها را از فایل JSON می‌خواند، برای اسناد embedding می‌سازد، سؤال کاربر را دسته‌بندی می‌کند و پاسخ را بر اساس اطلاعات واقعی رستوران تولید می‌کند.

برای ساخت این پروژه از ابزار های زیر استفاده شده است:

- Python
- FastAPI
- LangGraph
- OpenAI-compatible API
- Sentence Transformers
- Hybrid Search
- SQLite Conversation Memory
- Telegram Bot با `pyTelegramBotAPI`

## قابلیت‌ها

- پاسخ‌گویی درباره منو، مواد تشکیل‌دهنده، قیمت و وضعیت موجودی
- پاسخ‌گویی درباره آدرس، تلفن، ساعت کاری و اطلاعات عمومی رستوران
- تشخیص سؤال‌های عمومی درباره غذا
- جست‌وجوی ترکیبی معنایی و کلمه‌ای
- تحمل غلط‌های املایی فارسی در نام غذاها
- جست‌وجوی کامل catalog برای سؤال‌های فهرستی و مقایسه‌ای
- حالت مخصوص تحلیل اطلاعات غذا و کالری
- حافظه مکالمه با SQLite
- محدودکردن تعداد پیام‌های ذخیره‌شده برای هر session
- پاسخ‌دهی streaming در FastAPI و Telegram
- cache کردن مدل embedding و منابع RAG
- retry و مدیریت خطا برای درخواست‌های مدل
- ثبت زمان درخواست، زمان پاسخ، شناسه درخواست و token usage در logها
- جلوگیری از اجرای SQL مستقیم توسط مدل

## معماری کلی

جریان اصلی پاسخ‌گویی به شکل زیر است:

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

### مسیر سؤال رستورانی

1. سؤال با Router به یک `QuestionRoute` تبدیل می‌شود.
2. برنامه نوع عملیات را مشخص می‌کند: سؤال تکی، فهرست، مقایسه، فیلتر یا aggregate.
3. برای سؤال معمولی Hybrid Search با `TOP_K=3` اجرا می‌شود.
4. برای سؤال‌های فهرستی، مقایسه‌ای، فیلتر و aggregate کل catalog مرتبط بررسی می‌شود.
5. برای آدرس و اطلاعات تماس، سند اختصاصی `restaurant_information` مستقیماً خوانده می‌شود.
6. پروفایل ثابت رستوران همراه با اسناد مرتبط به مدل نهایی داده می‌شود.
7. پاسخ فقط بر اساس context معتبر تولید می‌شود.

## ساختار پروژه

```text
.
├── app/
│   ├── agent/
│   │   ├── agent_state.py       # State مشترک LangGraph
│   │   ├── router.py            # دسته‌بندی سؤال و ساخت query plan
│   │   ├── workflow.py          # ساخت و اجرای گراف LangGraph
│   │   └── workflow_nodes.py    # نودهای route، retrieve و generation
│   ├── core/
│   │   ├── embedding_model.py   # بارگذاری و cache مدل embedding
│   │   ├── llm_client.py        # کلاینت LLM، retry و logging
│   │   └── request_context.py   # شناسه درخواست برای logها
│   ├── memory/
│   │   └── conversation_memory.py # حافظه کوتاه‌مدت SQLite
│   ├── rag/
│   │   ├── rag_qa.py            # prompt و تولید پاسخ RAG
│   │   ├── retrieve.py          # نرمال‌سازی و جست‌وجوی Hybrid
│   │   └── retrieval_resources.py # cache مدل، اسناد و embeddingها
│   ├── telegram/
│   │   ├── bot.py               # اجرای ربات Telegram
│   │   ├── handlers.py          # handlerهای پیام و commandها
│   │   └── sender.py            # نمایش streaming پاسخ در Telegram
│   └── api.py                  # API اصلی FastAPI
├── data/
│   ├── offonburg_clean_rag.json       # منبع اصلی اطلاعات
│   ├── offonburg_documents.json       # اسناد تبدیل‌شده برای embedding
│   ├── offonburg_embedded_documents.json # اسناد همراه با index embedding
│   └── offonburg_embeddings.npy       # بردارهای embedding
├── scripts/
│   ├── rag_loader.py            # تبدیل JSON منبع به document
│   ├── embed_documents.py      # ساخت embeddingهای آفلاین
│   └── measure_api.py          # اندازه‌گیری زمان client و server
├── requirements.txt
├── .gitignore
└── README.md
```

## پیش‌نیازها

- Python 3.10 یا بالاتر؛ Python 3.12 پیشنهاد می‌شود.
- یک API Key از یک Provider سازگار با OpenAI Chat Completions
- برای ربات Telegram یک Bot Token
- مدل embedding دانلودشده یا دسترسی اولیه برای دانلود آن

## نصب

از ریشه پروژه اجرا کنید:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

در Windows فعال‌سازی محیط مجازی به شکل زیر است:

```powershell
.venv\Scripts\activate
```

## تنظیمات محیطی

فایل `.env` را در ریشه پروژه بسازید. این فایل نباید commit شود.

```env
API_KEY=your_provider_api_key
LLM_BASE_URL=https://your-provider.example/v1
LLM_MODEL_NAME=your-model-name
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_STREAM_EDIT_INTERVAL_SECONDS=0.8
MEMORY_MAX_MESSAGES=10
EMBEDDING_LOCAL_FILES_ONLY=true
```

نکات مهم:

- `LLM_MODEL_NAME` نام دقیق مدل در Provider است.
- `MEMORY_MAX_MESSAGES` تعداد پیام‌هاست، نه تعداد turnها؛ مقدار زوج استفاده کنید.
- اگر `EMBEDDING_LOCAL_FILES_ONLY=true` باشد، برنامه هنگام اجرا به اینترنت برای مدل embedding وصل نمی‌شود.

## آماده‌سازی داده‌های RAG

فایل منبع اصلی:

```text
data/offonburg_clean_rag.json
```

### اطلاعات رستوران

اطلاعات عمومی در بخش `restaurant` قرار می‌گیرد. فیلد `profile_text` برای متن آزاد درباره رستوران استفاده می‌شود:

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

### اطلاعات غذا

هر غذا باید در آرایه `menu` قرار بگیرد:

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

پس از هر تغییر در JSON، اسناد و embeddingها را دوباره بسازید:

```bash
./.venv/bin/python -m scripts.rag_loader
./.venv/bin/python -m scripts.embed_documents
```

از اجرای مستقیم `scripts/embed_documents.py` خودداری کنید؛ اجرای module با `-m` باعث می‌شود importهای پروژه به‌درستی کار کنند.

## اجرای API

از ریشه پروژه:

```bash
./.venv/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000
```

برای توسعه با reload:

```bash
./.venv/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

مدل embedding و اسناد RAG هنگام startup بارگذاری و در همان process cache می‌شوند.

## API Endpoints

### بررسی سلامت سرویس

```http
GET /health
```

پاسخ:

```json
{
  "status": "ok"
}
```

### دریافت پاسخ کامل

```http
POST /ask
Content-Type: application/json
```

بدنه درخواست:

```json
{
  "session_id": "user-123",
  "question": "همبرگرهای موجود را نام ببر"
}
```

نمونه پاسخ:

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

### پاسخ streaming با Server-Sent Events

```http
POST /ask/stream
Content-Type: application/json
```

اجرای آزمایشی با curl:

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"stream-test","question":"آدرس رستوران کجاست؟"}'
```

رویدادهای streaming شامل موارد زیر هستند:

- `start`: آغاز workflow
- `route`: مسیر انتخاب‌شده و query plan
- `sources`: metadata اسناد بازیابی‌شده
- `token`: قطعه‌ای از پاسخ تولیدشده
- `done`: پایان پاسخ
- `error`: خطای کنترل‌شده

## اجرای ربات Telegram

بعد از قرار دادن `TELEGRAM_BOT_TOKEN` در `.env` اجرا کنید:

```bash
./.venv/bin/python -m app.telegram.bot
```

ویژگی‌های ربات:

- استفاده از `pyTelegramBotAPI`
- ساخت خودکار `session_id` بر اساس شناسه chat و user
- جداکردن حافظه کاربران در گروه‌ها
- نمایش پاسخ به‌صورت تدریجی با ویرایش پیام Telegram
- تقسیم پاسخ‌های طولانی‌تر از محدودیت Telegram
- اجرای مشترک همان workflow استفاده‌شده توسط FastAPI

دستورهای موجود:

- `/start`
- `/help`
- ارسال پیام متنی عادی

## حافظه مکالمه

حافظه در فایل SQLite زیر ذخیره می‌شود:

```text
memory.db
```

برای هر `session_id`، پیام‌های user و assistant ذخیره می‌شوند. مقدار پیش‌فرض `MEMORY_MAX_MESSAGES=10` است؛ یعنی پنج turn کامل. پیام‌های قدیمی‌تر به‌صورت خودکار حذف می‌شوند.

در API مقدار `session_id` باید توسط client ارسال شود. در Telegram این مقدار به‌صورت خودکار ساخته می‌شود.

فایل `memory.db` داده runtime است و نباید در GitHub قرار بگیرد.

## جست‌وجوی Hybrid

جست‌وجوی معمولی از دو سیگنال استفاده می‌کند:

```text
hybrid_score = 0.7 * semantic_score + 0.3 * keyword_score
```

- بخش semantic با مدل `paraphrase-multilingual-MiniLM-L12-v2` محاسبه می‌شود.
- بخش keyword شامل token overlap و fuzzy matching نام غذا است.
- تفاوت حروف فارسی و عربی مانند `ي/ی`، `ك/ک` و `آ/ا` نرمال می‌شود.
- آستانه fuzzy matching نام غذا `0.82` است.
- برای سؤال‌های عادی فقط سه سند برتر برگردانده می‌شود.
- سؤال‌های list، comparison، filter و aggregate از catalog کامل مرتبط استفاده می‌کنند.

## Router و Query Plan

Router خروجی ساختاریافته زیر را تولید می‌کند:

- `category`: رستورانی، غذای عمومی، خارج از محدوده یا مبهم
- `operation`: سؤال تکی، فهرست، مقایسه، فیلتر یا aggregate
- `domain`: منو، مواد تشکیل‌دهنده، تغذیه، قیمت، موجودی یا اطلاعات رستوران
- `menu_category`: دسته منو در صورت تشخیص
- `target_item`: نام canonical غذا در صورت تشخیص
- `requires_full_catalog`: نیاز به بررسی کل catalog
- `calorie_mode`: عدم درخواست، فقط مقدار دقیق یا امکان تخمین

پاسخ Router قبل از استفاده با Pydantic اعتبارسنجی می‌شود. برای خروجی نامعتبر Router، سه retry با تأخیر کوتاه اجرا می‌شود.

## مدیریت خطا و logging

در `app/core/llm_client.py` برای خطاهای timeout، اتصال و rate limit، retry با exponential backoff وجود دارد. همچنین موارد زیر log می‌شوند:

- شروع و پایان درخواست LLM
- زمان پاسخ Provider
- شناسه درخواست
- تعداد tokenهای prompt و completion
- زمان بارگذاری embedding model
- زمان retrieval
- زمان بارگذاری و ذخیره حافظه
- زمان رسیدن اولین token در streaming

API این headerها را نیز برمی‌گرداند:

- `X-Request-ID`
- `X-Server-Time-Ms`

## اندازه‌گیری زمان API

ابتدا API را اجرا کنید و سپس در ترمینال دیگری بزنید:

```bash
./.venv/bin/python -m scripts.measure_api "آدرس رستوران کجاست؟"
```

گزینه‌های قابل استفاده:

```bash
./.venv/bin/python -m scripts.measure_api \
  --url http://127.0.0.1:8000/ask \
  --session-id timing-test \
  "قیمت آن پرایم چقدر است؟"
```

این ابزار زمان رفت‌وبرگشت client و زمان ثبت‌شده در server را مقایسه می‌کند.

## اجرای ابزارهای CLI داخلی

اجرای workflow از خط فرمان:

```bash
./.venv/bin/python -m app.agent.workflow
```

آزمایش retrieval:

```bash
./.venv/bin/python -m app.rag.retrieve
```

آزمایش پاسخ RAG:

```bash
./.venv/bin/python -m app.rag.rag_qa
```

## بررسی اولیه کد

برای بررسی syntax فایل‌های پروژه:

```bash
./.venv/bin/python -m compileall -q app scripts
```

برای اطمینان از سلامت داده‌های تولیدشده، این فایل‌ها باید با هم هماهنگ باشند:

- تعداد اسناد در `offonburg_embedded_documents.json`
- تعداد ردیف‌های `offonburg_embeddings.npy`
- ترتیب اسناد و embeddingها

بعد از تغییر متن یا منوی JSON، همیشه هر دو مرحله `rag_loader` و `embed_documents` را اجرا کنید.

## محدودیت‌های فعلی

- سیستم در حال حاضر ثبت سفارش و پرداخت آنلاین انجام نمی‌دهد.
- Tool Calling مربوط به سفارش‌ها در نسخه فعلی وجود ندارد.
- اطلاعات کالری در داده منبع کامل نیست؛ پاسخ دقیق فقط وقتی داده کالری موجود باشد ارائه می‌شود.
- به‌روزرسانی منو خودکار نیست و باید داده JSON و embeddingها دوباره ساخته شوند.

## مسیر پیشنهادی توسعه

1. افزودن تست‌های خودکار برای Router و retrieval
2. اضافه‌کردن ارزیابی کیفیت RAG با مجموعه سؤال و پاسخ مرجع
3. افزودن logging ساختاریافته و monitoring در محیط production
