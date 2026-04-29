# Submission Document — PDF-Constrained Conversational Agent

## 1. System Overview

**Project Name:** PDF Agent  
**Repository Name:** Stair_Digital

**Objective:**  
The system is a retrieval-grounded conversational assistant that answers user questions strictly from uploaded PDF content. It uses semantic retrieval, reranking, and strict citation validation before returning an answer. If evidence is weak or unavailable, it refuses instead of generating unsupported responses.

---

## 2. Architecture & Design Decisions

### 2.1 High-Level Architecture

Pipeline:

`PDF Input -> Parsing -> Cleaning -> Section Detection -> Chunking -> Embedding -> Vector Store -> Retrieval -> Hallucination Gate -> Reranking -> LLM -> Citation Validator -> Output`

Component summary:

- **PDF Parsing:** `PyMuPDF (fitz)` for structured text extraction from pages.
- **Chunking Strategy:** section-aware chunks with overlap (`512` tokens, `100` overlap) to preserve context.
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` for fast semantic retrieval.
- **Vector Store:** `ChromaDB` with cosine similarity space.
- **Retriever:** top-k retrieval (`k=10`) filtered by current `doc_id`.
- **Reranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2` reduces candidates to top `3`.
- **LLM:** Groq-hosted `llama-3.3-70b-versatile` at temperature `0.0`.
- **Grounding Layer:** pre-LLM retrieval gate + post-LLM citation validator to enforce PDF-only responses.

### 2.2 Key Design Decisions

| Decision | Choice | Reason | Trade-off |
|---|---|---|---|
| Chunk size | 512 tokens | Keeps enough context for semantic matching | Larger chunks can include noise |
| Chunk overlap | 100 tokens | Maintains continuity across chunk boundaries | Extra storage and near-duplicate text |
| Retrieval top-k | 10 | Better recall before reranking | More irrelevant chunks may enter first stage |
| Rerank top-k | 3 | Improves precision for final context | Risk of dropping less obvious relevant evidence |
| Citation format | Page + section/chunk identifiers | Human-verifiable references | Strict format can reject loosely formatted answers |

### 2.3 Hallucination Prevention Strategy

The system uses two barriers:

1. **Retrieval confidence gate (pre-LLM):**
   - If best similarity is below confidence threshold (distance above gate), the system refuses early.
2. **Citation validation gate (post-LLM):**
   - The answer must include valid citations mapped to retrieved chunk metadata.
   - If citations are missing/invalid, response is rejected.

Additionally:

- Prompt policy forbids outside knowledge.
- The model is instructed to return `[INSUFFICIENT]` when evidence is not present.
- Refusal template is returned when support is absent.

### 2.4 Refusal Policy

- **Out-of-PDF queries:** refuse.
- **Partial evidence:** answer only supported part and state limitations.
- **Ambiguous queries:** clarify through follow-up rewrite; refuse if still unsupported.
- **Low-confidence retrieval:** refuse before generation.

Standard refusal:

`This information is not available in the provided document.`

---

## 3. Observability & Debugging

The system provides internal visibility via UI trace and debug modules:

- Retrieved chunks are displayed with rank, page, section, and similarity diagnostics.
- Query trace includes:
  - original query
  - rewritten query (if applied)
  - gate decisions
  - response type (answer/refusal/error)
  - citation list
- Debug scripts are available for parser, pipeline, index, and retrieval verification.

Trace path:

`Query -> Retrieved Chunks -> Reranked Chunks -> Generated Answer -> Citation Validation -> Final Output`

---

## 4. Test Instructions (Evaluator Guide)

### Step 1: Setup

1. Install dependencies from `pdf_agent/requirements.txt`.
2. Create `.env` from `.env.example`.
3. Add `GROQ_API_KEY`.

### Step 2: Run Application

From repository root:

```bash
streamlit run pdf_agent/app.py
```

### Step 3: Upload PDF

Upload a PDF in the Streamlit UI and wait for ingestion/indexing completion.

### Step 4: Run Queries

Use the query list in Section 6 exactly.

### Step 5: Verify Output

Check:

- factual correctness from PDF
- citation mapping to actual page/section content
- refusal behavior for invalid/out-of-scope prompts
- retrieval trace visibility

---

## 5. Sample PDF Description

**Title:** RBI Monetary Policy Document (example)  
**Domain:** Financial policy/regulatory document

**Key Sections (example):**

- Section 1: Policy Rate Decisions
- Section 2: Growth and Inflation Projections

Note: repository currently does not bundle a single official sample PDF; evaluator can upload any suitable PDF.

---

## 6. Test Queries

### Valid Queries (Must Answer with Citations)

| Query | Expected Behavior | Expected Citation |
|---|---|---|
| What was the repo rate decided in the October 2023 meeting? | Returns exact policy decision from document | Page with policy decision section |
| What are the GDP growth projections for FY24? | Returns projection value(s) if present | Projection section/page |
| Explain the new gold loan rules for Urban Cooperative Banks. | Summarizes rule text from evidence | Relevant regulation section/page |
| What is the inflation outlook in this document? | Gives only grounded inflation statements | Inflation section/page |
| What is the policy stance mentioned? | Returns stance phrase exactly as stated | Policy stance section/page |

### Invalid / Out-of-Scope Queries

| Query | Expected Behavior |
|---|---|
| Who is the current Finance Minister of India? | Refusal |
| Compare this policy with US Federal Reserve policy. | Refusal |
| What did RBI say about crypto taxation? (if absent) | Refusal |

---

## 7. Example Outputs

### Example 1 (Valid)

**Query:** What was the repo rate decided in the October 2023 meeting?  
**Response:** The committee decided to keep the policy repo rate unchanged at X%, as stated in the uploaded document.  
**Citation:** Page X, Section Policy Rate Decision

### Example 2 (Invalid)

**Query:** Who is the current Finance Minister of India?  
**Response:** This information is not available in the provided document.

---

## 8. Limitations & Trade-offs

- OCR dependencies are present, but OCR flow is not fully integrated for all scanned PDFs.
- Approximate token counting is used in chunking, not tokenizer-exact counting.
- Large PDFs can reduce retrieval precision and increase ambiguity across similar sections.
- Table/image-heavy PDFs may not parse as accurately as plain text pages.
- Strict citation checks may occasionally reject otherwise useful responses when citation formatting drifts.

---

## 9. Deployment (Optional)

**Live URL:** Not deployed yet  
**Tech Stack:** Streamlit, ChromaDB, SentenceTransformers, CrossEncoder reranker, Groq LLM  
**Hosting:** Local-first (no production deployment config currently included)

---

## 10. Bonus Features

- Context-aware follow-up query rewriting for conversational continuity.
- Guarded context reuse to reduce topic drift in multi-turn chat.
- No explicit cross-language retrieval pipeline declared in current implementation.

---

## 11. Demo Video

**Link:** To be added (`Google Drive` or `YouTube`)

---

## Evaluation-Focused Notes

The implementation is strongest when demonstrating:

1. refusal under weak/out-of-scope evidence,
2. citation-to-content correctness,
3. retrieval trace visibility,
4. robust handling of tricky adversarial prompts.
