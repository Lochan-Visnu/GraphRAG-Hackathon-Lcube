# 🧬 MEDIC-AI: Tri-Pipeline GraphRAG Architecture
**A GraphRAG Inference Hackathon Submission by $L^3$ (Learn Lead Link)**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Gemini](https://img.shields.io/badge/Google-Gemini_2.5_Flash_Lite-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-green)
![TigerGraph](https://img.shields.io/badge/GraphDB-TigerGraph-red)

## 📌 Project Overview
The core objective of this project is to prove a single thesis: **Graphs make LLM inference faster, cheaper, and smarter than basic RAG alone.** To demonstrate this, $L^3$ engineered a concurrent, multi-threaded Tkinter dashboard that forces three distinct AI architectures to compete side-by-side on a massive medical dataset (2M+ tokens). The system tracks latency, token consumption, financial cost, and factual accuracy in real-time.

---

## 🚀 The Three Competing Pipelines

### 1️⃣ Pipeline 1: LLM-Only (Linear CSV Scan)
* **Mechanism:** Brute-force scans massive CSV matrices using Pandas to find keyword matches, passing raw chunks to the LLM.
* **Result:** Serves as the baseline. It suffers from high token bloat, slow latency, and significantly higher financial costs due to massive context windows.

### 2️⃣ Pipeline 2: Basic Vector RAG
* **Mechanism:** Uses `chromadb` and `sentence-transformers` to fetch semantically similar data chunks from the medical dataset.
* **Result:** Faster and cheaper than P1, but loses structural clinical context and critical relationships between specific medical conditions, symptoms, and treatments.

### 3️⃣ Pipeline 3: GraphRAG Topology (The Winner)
* **Mechanism:** Queries a local graph-extraction engine to pull explicit, mapped entity relationships (`Vertex`, `MANIFESTS_AS`, `MANAGED_BY`).
* **Result:** Delivers highly accurate clinical context with absolute minimal token usage. This drives down API costs while maximizing clinical safety and structural accuracy.

---

## ⚖️ Automated LLM-as-a-Judge Telemetry
To ensure an unbiased benchmark, the dashboard includes a live **LLM-as-a-Judge** scoring system. 
After a pipeline generates a response, the engine sends the summary back to the LLM to grade its factual accuracy and lack of hallucinations on a strict 1-10 scale. The judge's token usage is automatically rolled into the final cost metric for a 100% honest financial benchmark.

---

## ⚙️ Quick Start & Installation

### Prerequisites
* Windows Subsystem for Linux (WSL) installed.
* Docker Desktop running.
* Python 3.9 or higher.

### 1. Start the TigerGraph Database (Docker/WSL)
Open your WSL terminal and spin up the local container. We map the default gateway to `localhost:8000`:
```bash
docker run -d -p 8000:9000 -p 14240:14240 -p 9002:9002 --name tigergraph_medic_ai -v ~/data:/home/tigergraph/mydata -t [docker.tigergraph.com/tigergraph-dev:latest](https://docker.tigergraph.com/tigergraph-dev:latest)
