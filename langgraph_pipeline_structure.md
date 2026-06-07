# LangGraph Pipeline Structure

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────┐     retry (max 2x)
│  extract    │◄──────────────┐
│  (PDF→Prof) │───────────────┘
└──────┬──────┘
       │ continue
       ▼
┌─────────────┐
│   fetch     │
│ (CT.gov API)│
└──────┬──────┘
       │ continue
       ▼
┌─────────────┐
│   ingest    │
│(Embed→Qdrant│
└──────┬──────┘
       │ continue
       ▼
┌─────────────┐
│  retrieve   │
│(Semantic+   │
│ Rerank)     │
└──────┬──────┘
       │ continue
       ▼
┌─────────────┐
│   score     │
│(LLM Eligib.)│
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    END      │
└─────────────┘

* Any node error → END (fail)
```

## Node Summary

| Node | Input | Output | Model/Tool |
|------|-------|--------|------------|
| **extract** | PDF path | PatientProfile | Groq (Llama 3.3 70B) |
| **fetch** | diagnosis string | List[ClinicalTrial] | ClinicalTrials.gov API v2 |
| **ingest** | trials list | chunk count | PubMedBERT → Qdrant |
| **retrieve** | patient profile | top-8 chunks | Cosine search + LLM rerank |
| **score** | chunks + profile | scored trials | Groq (Llama 3.3 70B) |
