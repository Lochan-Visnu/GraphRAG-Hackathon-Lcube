import pandas as pd
import time
import tkinter as tk
from tkinter import ttk, scrolledtext
from google import genai
import threading

# ── 1. CONFIGURATION (STRICTLY UNCHANGED) ───────────────────────────
API_KEY = "AIzaSyCcQ2GGvWHpPbX4dFOI9pahHaNCVmIIKJI"
CSV_FILE = "medical_data_large.csv"
MODEL_ID = "gemini-2.5-flash-lite"

COST_PER_1M_INPUT = 0.075
COST_PER_1M_OUTPUT = 0.30

client = genai.Client(api_key=API_KEY)

def calculate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * COST_PER_1M_INPUT
    output_cost = (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT
    return input_cost + output_cost

# ── 2. ADVANCED UI CLASS ───────────────────────────────────────────
class PipelineProGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MEDIC-AI | Pipeline 1 Auditor")
        self.root.geometry("1000x750")
        self.root.configure(bg="#0F111A") # Deep Navy/Black

        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#0F111A")
        style.configure("Card.TFrame", background="#1A1C27", borderwidth=1)

    def setup_ui(self):
        # Navbar
        nav = tk.Frame(self.root, bg="#1A1C27", height=60)
        nav.pack(fill=tk.X)
        tk.Label(nav, text="🩺 MEDIC-AI SYSTEMS", bg="#1A1C27", fg="#569CD6", font=("Impact", 18)).pack(side=tk.LEFT, padx=20)
        tk.Label(nav, text="V1.0 | PIPELINE_1_MASSIVE_CSV", bg="#1A1C27", fg="#4EC9B0", font=("Consolas", 10)).pack(side=tk.RIGHT, padx=20)

        # Input Section
        input_container = tk.Frame(self.root, bg="#0F111A", pady=30)
        input_container.pack()
        
        self.query_entry = tk.Entry(input_container, width=50, font=("Segoe UI", 12), bg="#1A1C27", fg="#DCDCDC", 
                                    insertbackground="white", borderwidth=0, highlightthickness=1, highlightbackground="#3E4452")
        self.query_entry.pack(side=tk.LEFT, padx=10, ipady=8)
        self.query_entry.bind("<Return>", lambda e: self.start_analysis_thread())

        self.search_btn = tk.Button(input_container, text="START AUDIT", command=self.start_analysis_thread, 
                                   bg="#007ACC", fg="white", font=("Segoe UI Bold", 10), relief="flat", padx=25, pady=8, cursor="hand2")
        self.search_btn.pack(side=tk.LEFT)

        # Metrics Grid
        metrics_container = tk.Frame(self.root, bg="#0F111A")
        metrics_container.pack(pady=10)

        self.lat_box = self.create_stat_card(metrics_container, "TOTAL LATENCY", "0.00s", "#CE9178", 0)
        self.tok_box = self.create_stat_card(metrics_container, "TOKENS PROCESSED", "0", "#4FC1FF", 1)
        self.cst_box = self.create_stat_card(metrics_container, "ESTIMATED COST (USD)", "$0.000000", "#DCDCAA", 2)

        # Output Section
        output_frame = tk.Frame(self.root, bg="#0F111A", padx=40)
        output_frame.pack(fill=tk.BOTH, expand=True, pady=20)

        tk.Label(output_frame, text="TERMINAL OUTPUT", bg="#0F111A", fg="#808080", font=("Segoe UI Bold", 8)).pack(anchor="w")
        self.console = scrolledtext.ScrolledText(output_frame, bg="#050505", fg="#9CDCFE", font=("Consolas", 11), 
                                                borderwidth=0, highlightthickness=1, highlightbackground="#333")
        self.console.pack(fill=tk.BOTH, expand=True)

    def create_stat_card(self, parent, title, value, color, col):
        card = tk.Frame(parent, bg="#1A1C27", padx=25, pady=15, highlightthickness=1, highlightbackground="#333")
        card.grid(row=0, column=col, padx=10)
        tk.Label(card, text=title, bg="#1A1C27", fg="#808080", font=("Segoe UI", 8, "bold")).pack()
        lbl = tk.Label(card, text=value, bg="#1A1C27", fg=color, font=("Segoe UI Semibold", 16))
        lbl.pack()
        return lbl

    def write_console(self, msg, color="#9CDCFE"):
        self.console.insert(tk.END, f"> {msg}\n")
        self.console.see(tk.END)

    def start_analysis_thread(self):
        query = self.query_entry.get().strip()
        if not query: return
        threading.Thread(target=self.run_logic, args=(query,), daemon=True).start()

    def run_logic(self, query):
        self.search_btn.config(state="disabled", bg="#333")
        self.console.delete(1.0, tk.END)
        self.write_console(f"Initializing Auditor Session for [ {query} ]", "#4EC9B0")
        
        start_time = time.time()
        
        try:
            # ── RETRIEVAL (LINEAR SCAN) ──
            self.write_console("Linear Scan: Searching 1 Million rows in medical_data_large.csv...")
            found_rows = []
            for chunk in pd.read_csv(CSV_FILE, chunksize=100000):
                match = chunk[chunk.astype(str).apply(lambda x: x.str.contains(query, case=False)).any(axis=1)]
                if not match.empty:
                    found_rows.append(match.head(5))
                    if len(found_rows) > 3: break

            search_end = time.time()
            search_latency = round(search_end - start_time, 2)

            if not found_rows:
                self.write_console(f"Alert: Zero matches found for '{query}' in dataset.", "#F44747")
                self.search_btn.config(state="normal", bg="#007ACC")
                return

            context_text = pd.concat(found_rows).to_string(index=False)
            self.write_console(f"Retrieval Complete. Found relevant records in {search_latency}s.")

            # ── GENERATION (GEMINI) ──
            self.write_console(f"Model Dispatch: Forwarding context to {MODEL_ID}...")
            prompt = f"Using these medical records, provide a summary for {query}:\n\n{context_text}"
            
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            
            total_end = time.time()
            total_latency = round(total_end - start_time, 2)

            # ── METRICS ──
            in_tokens = response.usage_metadata.prompt_token_count
            out_tokens = response.usage_metadata.candidates_token_count
            total_tokens = response.usage_metadata.total_token_count
            query_cost = calculate_cost(in_tokens, out_tokens)

            # Thread-Safe UI Update
            self.root.after(0, self.update_final_ui, response.text, total_latency, total_tokens, query_cost)

        except Exception as e:
            self.write_console(f"CRITICAL SYSTEM ERROR: {str(e)}", "#F44747")
            self.search_btn.config(state="normal", bg="#007ACC")

    def update_final_ui(self, ai_text, latency, tokens, cost):
        self.write_console("── ANALYSIS REPORT GENERATED ──", "#4EC9B0")
        self.console.insert(tk.END, f"\n{ai_text}\n")
        self.lat_box.config(text=f"{latency}s")
        self.tok_box.config(text=f"{tokens:,}")
        self.cst_box.config(text=f"${cost:.6f}")
        self.search_btn.config(state="normal", bg="#007ACC")

if __name__ == "__main__":
    root = tk.Tk()
    app = PipelineProGUI(root)
    root.mainloop()
