#Pipeline 1 LLM only (non UI version)
import pandas as pd
import time
from google import genai

# ── 1. CONFIGURATION ────────────────────────────────
API_KEY = "YOUR_API_KEY" #Use gemini API
CSV_FILE = "medical_data_large.csv"
MODEL_ID = "gemini-2.5-flash-lite"

# PRICING (Hypothetical/Estimated for Gemini Flash Lite per 1M tokens)
COST_PER_1M_INPUT = 0.075  # $0.075 per million tokens
COST_PER_1M_OUTPUT = 0.30  # $0.30 per million tokens

client = genai.Client(api_key=API_KEY)

def calculate_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1_000_000) * COST_PER_1M_INPUT
    output_cost = (output_tokens / 1_000_000) * COST_PER_1M_OUTPUT
    return input_cost + output_cost

def run_p1():
    print("\n" + "🩺 " + "="*25 + " PIPELINE 1: MASSIVE CSV " + "="*25)
    query = input("Search Disease/Symptom in 1M Rows: ").strip()
    if not query: return

    # --- START TIMER ---
    start_time = time.time()
    
    # ── 2. RETRIEVAL (LINEAR SCAN) ──────────────────
    print(f"🔍 [RETRIEVAL] Scanning 987,653 rows for '{query}'...")
    try:
        found_rows = []
        # Chunking simulates the heavy disk I/O of traditional data systems
        for chunk in pd.read_csv(CSV_FILE, chunksize=100000):
            match = chunk[chunk.astype(str).apply(lambda x: x.str.contains(query, case=False)).any(axis=1)]
            if not match.empty:
                found_rows.append(match.head(5))
                if len(found_rows) > 3: break # Enough context

        search_end = time.time()
        search_latency = round(search_end - start_time, 2)

        if not found_rows:
            print(f"⚠️ No records found for '{query}' in the dataset.")
            return

        context_text = pd.concat(found_rows).to_string(index=False)

        # ── 3. GENERATION (GEMINI) ────────────────────
        print(f"🧠 [GENERATION] Using {MODEL_ID}...")
        prompt = f"Using these medical records, provide a summary for {query}:\n\n{context_text}"
        
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt
        )
        
        total_end = time.time()
        gen_latency = round(total_end - search_end, 2)
        total_latency = round(total_end - start_time, 2)

        # ── 4. METRICS & COST CALCULATION ─────────────
        # Extract token counts from Gemini metadata
        in_tokens = response.usage_metadata.prompt_token_count
        out_tokens = response.usage_metadata.candidates_token_count
        total_tokens = response.usage_metadata.total_token_count
        
        query_cost = calculate_cost(in_tokens, out_tokens)

        # ── 5. FINAL REPORT ───────────────────────────
        print("\n" + "✨ GEMINI'S ANALYSIS:")
        print(response.text)
        
        print("\n" + "📊 DEMO PERFORMANCE METRICS:")
        print(f"┃  Total Latency:    {total_latency}s")
        print(f"┃  ┣━ Retrieval:     {search_latency}s (Linear CSV Scan)")
        print(f"┃  ┗━ AI Generation: {gen_latency}s")
        print(f"┃")
        print(f"┃  Token Usage:      {total_tokens} tokens")
        print(f"┃  ┣━ Input/Context: {in_tokens}")
        print(f"┃  ┗━ Output/Answer: {out_tokens}")
        print(f"┃")
        print(f"┃  Estimated Cost:   ${query_cost:.8f}")
        print("━"*70)

    except Exception as e:
        print(f"❌ System Error: {e}")

if __name__ == "__main__":
    print(f"🚀 Pipeline 1 Online | Model: {MODEL_ID} | Dataset: 1 Million Rows")
    while True:
        run_p1()
