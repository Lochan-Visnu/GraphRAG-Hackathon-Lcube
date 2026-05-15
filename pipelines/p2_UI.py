import time
import os
import re
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import errors

# ── CONFIG (Identical to yours) ────────────────────────────────────────
API_KEY  = "AIzaSyCcQ2GGvWHpPbX4dFOI9pahHaNCVmIIKJI"
MODEL_ID = "gemini-2.5-flash-lite"
DATASET  = r"C:\Users\LIkhitha\graphrag_project\data2m.txt"
DB_PATH  = r"C:\Users\LIkhitha\graphrag_project\chroma_db_v2"

# ── INIT (Identical to yours) ──────────────────────────────────────────
client     = genai.Client(api_key=API_KEY)
embedder   = SentenceTransformer("all-MiniLM-L6-v2")
chroma     = chromadb.PersistentClient(path=DB_PATH)
collection = chroma.get_or_create_collection("medical_v2")

# ── SMART CHUNKING (Identical logic) ──────────────────────────────────
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
            if len(para) <= 1200: chunks.append(para)
            else:
                words = para.split()
                for i in range(0, len(words), 300):
                    chunk = " ".join(words[i:i+300])
                    if len(chunk) > 80: chunks.append(chunk)
    return chunks

# ── GUI CLASS ──────────────────────────────────────────────────────────
class RAG_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MEDIC-AI | Pipeline 2: Basic RAG")
        self.root.geometry("1000x800")
        self.root.configure(bg="#0F111A")

        self.setup_ui()
        
        # Check if we need to index on startup
        if collection.count() == 0:
            threading.Thread(target=self.run_indexing, daemon=True).start()
        else:
            self.log(f"✅ ChromaDB ready — {collection.count():,} chunks")

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#1A1C27", height=70)
        header.pack(fill=tk.X)
        tk.Label(header, text="🔍 PIPELINE 2: BASIC RAG", bg="#1A1C27", fg="#4EC9B0", font=("Impact", 20)).pack(side=tk.LEFT, padx=30)
        tk.Label(header, text="VECTOR STORAGE + GEMINI LITE", bg="#1A1C27", fg="#888", font=("Consolas", 10)).pack(side=tk.RIGHT, padx=30)

        # Metrics Row
        metric_frame = tk.Frame(self.root, bg="#0F111A")
        metric_frame.pack(fill=tk.X, padx=30, pady=20)
        
        self.lat_box = self.create_metric(metric_frame, "LATENCY", "0.00s", "#CE9178", 0)
        self.tok_box = self.create_metric(metric_frame, "TOKENS", "0", "#4FC1FF", 1)
        self.cst_box = self.create_metric(metric_frame, "COST", "$0.0000", "#DCDCAA", 2)

        # Search Bar
        search_frame = tk.Frame(self.root, bg="#0F111A")
        search_frame.pack(fill=tk.X, padx=30)
        
        self.q_entry = tk.Entry(search_frame, font=("Segoe UI", 12), bg="#1A1C27", fg="white", borderwidth=0, highlightthickness=1, highlightbackground="#333")
        self.q_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=(0, 10))
        self.q_entry.bind("<Return>", lambda e: self.start_ask_thread())

        self.ask_btn = tk.Button(search_frame, text="ASK SYSTEM", command=self.start_ask_thread, bg="#4EC9B0", fg="#000", font=("Segoe UI Bold", 10), relief="flat", padx=20)
        self.ask_btn.pack(side=tk.RIGHT)

        # Output Area
        tk.Label(self.root, text="SYSTEM TERMINAL / ANSWER OUTPUT", bg="#0F111A", fg="#555", font=("Segoe UI Bold", 8)).pack(anchor="w", padx=30, pady=(20, 5))
        self.console = scrolledtext.ScrolledText(self.root, bg="#050505", fg="#9CDCFE", font=("Consolas", 11), borderwidth=0, highlightthickness=1, highlightbackground="#333")
        self.console.pack(fill=tk.BOTH, expand=True, padx=30, pady=(0, 30))

    def create_metric(self, parent, label, value, color, col):
        card = tk.Frame(parent, bg="#1A1C27", padx=20, pady=10, highlightthickness=1, highlightbackground="#333")
        card.grid(row=0, column=col, padx=5, sticky="nsew")
        parent.grid_columnconfigure(col, weight=1)
        tk.Label(card, text=label, bg="#1A1C27", fg="#888", font=("Segoe UI Bold", 8)).pack()
        lbl = tk.Label(card, text=value, bg="#1A1C27", fg=color, font=("Segoe UI Semibold", 16))
        lbl.pack()
        return lbl

    def log(self, text, color="#9CDCFE"):
        self.console.insert(tk.END, f"> {text}\n")
        self.console.see(tk.END)

    def start_ask_thread(self):
        q = self.q_entry.get().strip()
        if not q: return
        self.ask_btn.config(state="disabled")
        threading.Thread(target=self.run_rag_logic, args=(q,), daemon=True).start()

    # ── LOGIC INTEGRATION (Strictly keeping your logic) ──────────────
    def run_indexing(self):
        self.log("📚 Indexing dataset... please wait.")
        if not os.path.exists(DATASET):
            self.log(f"❌ File not found: {DATASET}", "#F44747")
            return
        
        with open(DATASET, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        chunks = smart_chunk(text)
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            embeddings = embedder.encode(batch, show_progress_bar=False).tolist()
            ids = [f"chunk_{i+j}" for j in range(len(batch))]
            try: collection.add(documents=batch, embeddings=embeddings, ids=ids)
            except: pass
        self.log(f"✅ Indexed {collection.count():,} chunks")

    def run_rag_logic(self, question):
        try:
            self.console.delete(1.0, tk.END)
            self.log(f"Question: {question}", "#4EC9B0")
            
            # Retrieval Step
            q_embed = embedder.encode([question]).tolist()
            results = collection.query(query_embeddings=q_embed, n_results=3)
            
            self.log("🔎 Top Context Retrieved...")
            for i, doc in enumerate(results["documents"][0]):
                self.log(f"[{i+1}] {doc[:100]}...")

            context = "\n\n".join(results["documents"][0])
            prompt = f"Use the medical context below to answer the question. Be concise.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"

            # Generation Step
            start = time.time()
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            latency = round(time.time() - start, 2)

            # Metrics
            total_tokens = response.usage_metadata.total_token_count
            cost = round((total_tokens / 1_000_000) * 0.075, 6)

            # Update UI
            self.root.after(0, self.update_results, response.text, latency, total_tokens, cost)

        except Exception as e:
            self.log(f"❌ Error: {str(e)}", "#F44747")
            self.ask_btn.config(state="normal")

    def update_results(self, answer, lat, tok, cost):
        self.log("\n📝 FINAL ANSWER:")
        self.console.insert(tk.END, f"{answer}\n", "white")
        self.lat_box.config(text=f"{lat}s")
        self.tok_box.config(text=f"{tok:,}")
        self.cst_box.config(text=f"${cost}")
        self.ask_btn.config(state="normal")
        self.q_entry.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    # Define a simple tag for white text
    app = RAG_GUI(root)
    app.console.tag_config("white", foreground="#FFFFFF")
    root.mainloop()
