import time
import re
import os
import json
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk, messagebox
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

# ── 1. GLOBAL ENVIRONMENT CONFIGURATION ────────────────────────────────
API_KEY = "AIzaSyA9ScdyZHPAt5tDeho9vuplJVYtAQMo0NA"
MODEL_ID = "gemini-2.5-flash-lite"

# Shared Medical Asset Paths
CSV_FILE = r"C:\Users\Likhitha\graphrag_project\medical_data_fixed.csv"
DATASET_TXT = r"C:\Users\Likhitha\graphrag_project\data2m.txt"
DB_PATH = r"C:\Users\Likhitha\graphrag_project\chroma_db_v2"
GRAPH_NAME = "MedicalGraphRAG"

# Financial Free Tier Estimates per 1M Tokens
COST_PER_1M_INPUT = 0.075
COST_PER_1M_OUTPUT = 0.30

# Initialize Global Shared Controllers
print("🚀 Booting Global AI Clients & Vector Databases...")
try:
    client = genai.Client(api_key=API_KEY)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    chroma = chromadb.PersistentClient(path=DB_PATH)
    collection = chroma.get_or_create_collection("medical_v2")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

def calculate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * COST_PER_1M_INPUT
    output_cost = (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT
    return input_cost + output_cost

# ── 2. DATA PROCESSING ENGINES (P2 & P3) ───────────────────────────────
def smart_chunk(text):
    chunks = []
    qa_blocks = re.split(r'(?=###\s*Instruction|###\s*Question|Q:|Question:)', text)
    if len(qa_blocks) > 100:
        for block in qa_blocks:
            block = block.strip()
            if len(block) > 80: chunks.append(block[:1200])
    else:
        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]
        for para in paragraphs:
            if len(para) <= 1200: 
                chunks.append(para)
            else:
                words = para.split()
                for i in range(0, len(words), 300):
                    chunk = " ".join(words[i:i+300])
                    if len(chunk) > 80: chunks.append(chunk)
    return chunks

class KnowledgeGraphRAGEngine:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
    def query_subgraph_matrix(self, query_string):
        if not os.path.exists(self.dataset_path): 
            return None
        target = query_string.lower()
        edges, nodes = [], set()
        
        try:
            for chunk in pd.read_csv(self.dataset_path, chunksize=50000, dtype=str, keep_default_na=False):
                matched_rows = chunk[chunk.apply(lambda row: row.astype(str).str.contains(target, case=False).any(), axis=1)]
                for _, row in matched_rows.head(3).iterrows():
                    cols = list(chunk.columns)
                    disease = str(row.get('DISEASE', row.iloc[0] if len(cols) > 0 else query_string.upper())).strip()
                    symptoms = str(row.get('SYMPTOMS', row.iloc[1] if len(cols) > 1 else '')).strip()
                    treatments = str(row.get('CURE / TREATMENT', row.iloc[2] if len(cols) > 2 else '')).strip()
                    
                    if not disease or disease.lower() == 'nan': 
                        disease = query_string.upper()
                    nodes.add(f"Vertex(Type='Disease', ID='{disease}')")
                    
                    if symptoms and symptoms.lower() != 'nan':
                        sym_id = symptoms[:25] + "..."
                        nodes.add(f"Vertex(Type='SymptomCluster', ID='{sym_id}')")
                        edges.append({"source": disease, "edge_type": "MANIFESTS_AS", "target": sym_id, "metadata": {"raw": symptoms}})
                        
                    if treatments and treatments.lower() != 'nan':
                        treat_id = treatments[:25] + "..."
                        nodes.add(f"Vertex(Type='TherapeuticProtocol', ID='{treat_id}')")
                        edges.append({"source": disease, "edge_type": "MANAGED_BY", "target": treat_id, "metadata": {"raw": treatments}})
                if len(edges) >= 4: 
                    break
        except Exception: 
            pass
            
        if not edges: 
            return {"status": "EMPTY", "extracted_nodes": [query_string.upper()], "relationships": []}
        return {"status": "SUCCESS", "nodes": list(nodes), "relationships": edges}

graph_engine = KnowledgeGraphRAGEngine(CSV_FILE)

# ── 3. CORE INTERFACE ARCHITECTURE ────────────────────────────────────
class SystemDuelDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("MEDIC-AI | Tri-Pipeline Architecture")
        self.root.geometry("1600x900")
        self.root.configure(bg="#0F111A")
        
        self.setup_ui()
        
        # Passive Check for Vector DB
        if collection.count() == 0:
            db_thread = threading.Thread(target=self.bootstrap_vector_db, daemon=True)
            db_thread.start()
        else:
            self.write_log(self.p2_console, f"📦 ChromaDB Engine Ready: {collection.count():,} active vectors.")

    def setup_ui(self):
        # Navigation Bar
        nav = tk.Frame(self.root, bg="#1A1C27", height=65)
        nav.pack(fill=tk.X)
        
        lbl_main = tk.Label(
            nav, text="🧬 MEDIC-AI Comparison", 
            bg="#1A1C27", fg="#4EC9B0", font=("Impact", 20)
        )
        lbl_main.pack(side=tk.LEFT, padx=30, pady=15)
        
        lbl_sub = tk.Label(
            nav, text="3-WAY ARCHITECTURE BENCHMARK", 
            bg="#1A1C27", fg="#888", font=("Consolas", 10)
        )
        lbl_sub.pack(side=tk.RIGHT, padx=30)

        # Control Query Panel
        control_panel = tk.Frame(self.root, bg="#0F111A", pady=20)
        control_panel.pack(fill=tk.X, padx=30)
        
        lbl_input = tk.Label(
            control_panel, text="Global Query Input:", 
            bg="#0F111A", fg="#DCDCDC", font=("Segoe UI Semibold", 11)
        )
        lbl_input.pack(side=tk.LEFT, padx=(0, 10))
        
        self.query_entry = tk.Entry(
            control_panel, font=("Segoe UI", 12), bg="#1A1C27", 
            fg="white", borderwidth=0, highlightthickness=1, 
            highlightbackground="#333"
        )
        self.query_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 15))
        self.query_entry.bind("<Return>", lambda e: self.trigger_system_duel())

        self.duel_btn = tk.Button(
            control_panel, text="EXECUTE TRI-PIPELINE", 
            command=self.trigger_system_duel, bg="#007ACC", 
            fg="white", font=("Segoe UI Bold", 10), 
            relief="flat", padx=25, cursor="hand2"
        )
        self.duel_btn.pack(side=tk.RIGHT, ipady=4)

        # Main Split Frame Workspace Layout (3 Columns)
        split_workspace = tk.Frame(self.root, bg="#0F111A")
        split_workspace.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        for i in range(3): 
            split_workspace.columnconfigure(i, weight=1)

        # ==== LEFT: PIPELINE 1 (LLM ONLY / CSV) ====
        p1_frame = tk.Frame(split_workspace, bg="#131520", highlightthickness=1, highlightbackground="#222")
        p1_frame.grid(row=0, column=0, sticky="nsew", padx=10)
        
        lbl_p1 = tk.Label(p1_frame, text="📋 P1: LLM ONLY", bg="#1A1C27", fg="#CE9178", font=("Segoe UI Bold", 11), pady=6)
        lbl_p1.pack(fill=tk.X)
        
        p1_metrics = tk.Frame(p1_frame, bg="#131520", pady=10)
        p1_metrics.pack(fill=tk.X)
        self.p1_lat = self.create_metric_node(p1_metrics, "LATENCY", "0.00s", "#CE9178", 0)
        self.p1_tok = self.create_metric_node(p1_metrics, "TOKENS", "0", "#4FC1FF", 1)
        self.p1_cst = self.create_metric_node(p1_metrics, "COST (USD)", "$0.000", "#DCDCAA", 2)
        self.p1_jdg = self.create_metric_node(p1_metrics, "LLM JUDGE", "0/10", "#C586C0", 3)
        
        self.p1_console = scrolledtext.ScrolledText(p1_frame, bg="#050505", fg="#CE9178", font=("Consolas", 10), borderwidth=0)
        self.p1_console.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ==== CENTER: PIPELINE 2 (VECTOR DB) ====
        p2_frame = tk.Frame(split_workspace, bg="#131520", highlightthickness=1, highlightbackground="#222")
        p2_frame.grid(row=0, column=1, sticky="nsew", padx=10)
        
        lbl_p2 = tk.Label(p2_frame, text="🔍 P2: BASIC VECTOR RAG", bg="#1A1C27", fg="#4EC9B0", font=("Segoe UI Bold", 11), pady=6)
        lbl_p2.pack(fill=tk.X)
        
        p2_metrics = tk.Frame(p2_frame, bg="#131520", pady=10)
        p2_metrics.pack(fill=tk.X)
        self.p2_lat = self.create_metric_node(p2_metrics, "LATENCY", "0.00s", "#4EC9B0", 0)
        self.p2_tok = self.create_metric_node(p2_metrics, "TOKENS", "0", "#4FC1FF", 1)
        self.p2_cst = self.create_metric_node(p2_metrics, "COST (USD)", "$0.000", "#DCDCAA", 2)
        self.p2_jdg = self.create_metric_node(p2_metrics, "LLM JUDGE", "0/10", "#C586C0", 3)
        
        self.p2_console = scrolledtext.ScrolledText(p2_frame, bg="#050505", fg="#9CDCFE", font=("Consolas", 10), borderwidth=0)
        self.p2_console.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ==== RIGHT: PIPELINE 3 (GRAPHRAG) ====
        p3_frame = tk.Frame(split_workspace, bg="#131520", highlightthickness=1, highlightbackground="#222")
        p3_frame.grid(row=0, column=2, sticky="nsew", padx=10)
        
        lbl_p3 = tk.Label(p3_frame, text="🕸️ P3: TG GRAPHRAG TOPOLOGY", bg="#1A1C27", fg="#E2B93D", font=("Segoe UI Bold", 11), pady=6)
        lbl_p3.pack(fill=tk.X)
        
        p3_metrics = tk.Frame(p3_frame, bg="#131520", pady=10)
        p3_metrics.pack(fill=tk.X)
        self.p3_lat = self.create_metric_node(p3_metrics, "LATENCY", "0.00s", "#E2B93D", 0)
        self.p3_tok = self.create_metric_node(p3_metrics, "TOKENS", "0", "#4FC1FF", 1)
        self.p3_cst = self.create_metric_node(p3_metrics, "COST (USD)", "$0.000", "#DCDCAA", 2)
        self.p3_jdg = self.create_metric_node(p3_metrics, "LLM JUDGE", "0/10", "#C586C0", 3)
        
        self.p3_console = scrolledtext.ScrolledText(p3_frame, bg="#050505", fg="#DCDCAA", font=("Consolas", 10), borderwidth=0)
        self.p3_console.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_metric_node(self, parent, title, value, color, col):
        box = tk.Frame(parent, bg="#1A1C27", padx=8, pady=8, highlightthickness=1, highlightbackground="#2D3139")
        box.grid(row=0, column=col, padx=4, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)
        
        tk.Label(box, text=title, bg="#1A1C27", fg="#6A737D", font=("Segoe UI Bold", 7)).pack()
        lbl = tk.Label(box, text=value, bg="#1A1C27", fg=color, font=("Consolas", 11, "bold"))
        lbl.pack()
        return lbl

    def write_log(self, console, text):
        console.insert(tk.END, f"> {text}\n")
        console.see(tk.END)

    def bootstrap_vector_db(self):
        self.write_log(self.p2_console, "⚠️ ChromaDB empty — Bootstrapping from text files...")
        if not os.path.exists(DATASET_TXT):
            self.write_log(self.p2_console, f"❌ Aborted: Dataset path not found: {DATASET_TXT}")
            return
            
        with open(DATASET_TXT, "r", encoding="utf-8", errors="ignore") as f: 
            text = f.read()
            
        chunks = smart_chunk(text)
        for i in range(0, len(chunks), 32):
            batch = chunks[i:i+32]
            embeddings = embedder.encode(batch, show_progress_bar=False).tolist()
            try: 
                ids_list = [f"chk_{i+j}" for j in range(len(batch))]
                collection.add(documents=batch, embeddings=embeddings, ids=ids_list)
            except: 
                pass
        self.write_log(self.p2_console, f"✅ Bootstrapping complete! Total vectors: {collection.count():,}")

    # ── LLM-AS-A-JUDGE SCORING FUNCTION ───────────────────────────────
    def evaluate_with_llm_judge(self, query, context, generated_answer):
        judge_prompt = (
            f"You are an expert medical AI judge. Evaluate the generated answer below based on the provided context.\n"
            f"Score it strictly from 1 to 10 based on factual accuracy, relevance to the query, and lack of hallucination.\n"
            f"RESPOND ONLY WITH A SINGLE INTEGER FROM 1 TO 10.\n\n"
            f"Query: {query}\n"
            f"Context: {context}\n"
            f"Answer: {generated_answer}"
        )
        try:
            j_response = client.models.generate_content(model=MODEL_ID, contents=judge_prompt)
            in_t = j_response.usage_metadata.prompt_token_count
            out_t = j_response.usage_metadata.candidates_token_count
            
            # Extract just the digit from the response
            match = re.search(r'\d+', j_response.text.strip())
            score = match.group() if match else "N/A"
            return f"{score}/10", in_t, out_t
        except Exception:
            return "ERR", 0, 0

    def trigger_system_duel(self):
        query = self.query_entry.get().strip()
        if not query: 
            return
        
        self.duel_btn.config(state="disabled", bg="#2D3139")
        for console in [self.p1_console, self.p2_console, self.p3_console]: 
            console.delete(1.0, tk.END)
        
        # Fire all 3 engines concurrently
        t1 = threading.Thread(target=self.execute_pipeline_1, args=(query,), daemon=True)
        t2 = threading.Thread(target=self.execute_pipeline_2, args=(query,), daemon=True)
        t3 = threading.Thread(target=self.execute_pipeline_3, args=(query,), daemon=True)
        
        t1.start()
        t2.start()
        t3.start()

    # ── PIPELINE 1: MASSIVE CSV LINEAR SCAN ───────────────────────────
    def execute_pipeline_1(self, query):
        self.write_log(self.p1_console, f"Initializing Linear Scan for: '{query}'")
        start_time = time.time()
        try:
            found_rows = []
            for chunk in pd.read_csv(CSV_FILE, chunksize=50000, dtype=str, keep_default_na=False):
                match = chunk[chunk.apply(lambda row: row.astype(str).str.contains(query, case=False).any(), axis=1)]
                if not match.empty:
                    found_rows.append(match.head(5))
                    if len(found_rows) > 3: 
                        break

            if not found_rows:
                self.write_log(self.p1_console, f"⚠️ No matches located.")
                self.root.after(0, lambda: self.duel_btn.config(state="normal", bg="#007ACC"))
                return

            context_text = pd.concat(found_rows).to_string(index=False)
            prompt = f"Using these messy records, summarize {query}:\n\n{context_text}"
            
            # Step 1: Generate Answer
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            in_t = response.usage_metadata.prompt_token_count
            out_t = response.usage_metadata.candidates_token_count
            
            # Step 2: LLM-as-a-Judge Evaluation
            self.write_log(self.p1_console, "Running LLM Judge Evaluation...")
            judge_score, j_in, j_out = self.evaluate_with_llm_judge(query, context_text, response.text)
            
            # Combine metrics
            latency = round(time.time() - start_time, 2)
            total_in = in_t + j_in
            total_out = out_t + j_out
            cost = calculate_cost(total_in, total_out)
            
            self.root.after(
                0, self.update_ui, self.p1_console, self.p1_lat, 
                self.p1_tok, self.p1_cst, self.p1_jdg, response.text, 
                latency, total_in + total_out, cost, judge_score
            )
        except Exception as e:
            self.write_log(self.p1_console, f"❌ P1 Error: {e}")
            self.root.after(0, lambda: self.duel_btn.config(state="normal", bg="#007ACC"))

    # ── PIPELINE 2: BASIC VECTOR RAG ──────────────────────────────────
    def execute_pipeline_2(self, query):
        self.write_log(self.p2_console, f"Initializing Vector Lookup for: '{query}'")
        start_time = time.time()
        try:
            q_embed = embedder.encode([query]).tolist()
            results = collection.query(query_embeddings=q_embed, n_results=3)
            context = "\n\n".join(results["documents"][0]) if results["documents"] else ""
            
            prompt = f"Use this context to answer regarding {query}:\n\n{context}\n\nANSWER:"
            
            # Step 1: Generate Answer
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            in_t = response.usage_metadata.prompt_token_count
            out_t = response.usage_metadata.candidates_token_count
            
            # Step 2: LLM-as-a-Judge Evaluation
            self.write_log(self.p2_console, "Running LLM Judge Evaluation...")
            judge_score, j_in, j_out = self.evaluate_with_llm_judge(query, context, response.text)
            
            # Combine metrics
            latency = round(time.time() - start_time, 2)
            total_in = in_t + j_in
            total_out = out_t + j_out
            cost = calculate_cost(total_in, total_out)
            
            self.root.after(
                0, self.update_ui, self.p2_console, self.p2_lat, 
                self.p2_tok, self.p2_cst, self.p2_jdg, response.text, 
                latency, total_in + total_out, cost, judge_score
            )
        except Exception as e:
            self.write_log(self.p2_console, f"❌ P2 Error: {e}")

    # ── PIPELINE 3: GRAPHRAG TOPOLOGY ─────────────────────────────────
    def execute_pipeline_3(self, query):
        self.write_log(self.p3_console, f"Traversing Knowledge Graph topology for: '{query}'")
        start_time = time.time()
        try:
            graph_facts = graph_engine.query_subgraph_matrix(query)
            facts_context = json.dumps(graph_facts, indent=2)
            
            prompt = (
                f"You are a Clinical Presenter. Review the provided JSON sub-graph data and compress it into a highly scannable summary.\n"
                f"CRITICAL RULES: DO NOT quote raw json. Summarize clinical meaning using bullet points.\n\n"
                f"LIVE KNOWLEDGE GRAPH CONTEXT:\n{facts_context}\n\n"
                f"USER ENQUIRY: {query}.\n\nOUTPUT FORMAT:\n**Verdict:** [1 sentence]\n**Manifestations:**\n* [Bullets]\n**Management:**\n* [Bullets]"
            )
            
            # Step 1: Generate Answer
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            in_t = response.usage_metadata.prompt_token_count
            out_t = response.usage_metadata.candidates_token_count
            
            # Step 2: LLM-as-a-Judge Evaluation
            self.write_log(self.p3_console, "Running LLM Judge Evaluation...")
            judge_score, j_in, j_out = self.evaluate_with_llm_judge(query, facts_context, response.text)
            
            # Combine metrics
            latency = round(time.time() - start_time, 2)
            total_in = in_t + j_in
            total_out = out_t + j_out
            cost = calculate_cost(total_in, total_out)
            
            self.root.after(
                0, self.update_ui, self.p3_console, self.p3_lat, 
                self.p3_tok, self.p3_cst, self.p3_jdg, response.text, 
                latency, total_in + total_out, cost, judge_score
            )
        except Exception as e:
            self.write_log(self.p3_console, f"❌ P3 Error: {e}")

    def update_ui(self, console, l_lat, l_tok, l_cst, l_jdg, answer, latency, tokens, cost, judge_score):
        self.write_log(console, "\n✨ ANALYSIS REPORT:")
        console.insert(tk.END, f"\n{answer}\n")
        l_lat.config(text=f"{latency}s")
        l_tok.config(text=f"{tokens:,}")
        l_cst.config(text=f"${cost:.5f}")
        l_jdg.config(text=judge_score)
        
        self.duel_btn.config(state="normal", bg="#007ACC")

# ── 4. RUN MATRIX EXECUTION ───────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = SystemDuelDashboard(root)
    root.mainloop()
