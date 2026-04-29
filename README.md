# Yashraj — IDSA Streamlit UI

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What This Covers

| Step | What happens |
|------|-------------|
| **Step 1** | Client fills in Business Name, Industry, Daily Users, Data Setup, Goals → stored as Python dict |
| **Step 2** | 5 AI-generated questions appear (mocked now → replace with Dhrupad's Cortex call) |
| **Step 3** | User answers each question in text inputs → triggers RAG + LLM (Yagyansh + Dhrupad) |
| **Step 4** | Final insights displayed in a styled output card + raw JSON viewer |

## Where to Plug In Real Logic

In `app.py`, replace these two mock functions:

```python
def mock_generate_questions(client_data):
    # → Call Dhrupad's Snowflake Cortex function
    pass

def mock_generate_insights(client_data, qa_pairs):
    # → Call Yagyansh's RAG retrieval + Dhrupad's Cortex LLM
    pass
```

## Snowflake Tables Used (Bhumika's setup)
- `advisor_db.client_details` — stores client input
- `advisor_db.questions` — stores generated questions
- `advisor_db.answers` — stores Q&A pairs
- `advisor_db.documents_chunks` — used by RAG (Yagyansh)
