import time

import os

import chromadb

from sentence_transformers import SentenceTransformer

from google import genai

from google.genai import errors



# ── CONFIG ────────────────────────────────────────

API_KEY  = "AIzaSyCcQ2GGvWHpPbX4dFOI9pahHaNCVmIIKJI"

MODEL_ID = "gemini-2.5-flash-lite"

DATASET  = r"C:\Users\LIkhitha\graphrag_project\data2m.txt"

DB_PATH  = r"C:\Users\LIkhitha\graphrag_project\chroma_db_v2"  # fresh DB



# ── INIT ──────────────────────────────────────────

client     = genai.Client(api_key=API_KEY)

embedder   = SentenceTransformer("all-MiniLM-L6-v2")

chroma     = chromadb.PersistentClient(path=DB_PATH)

collection = chroma.get_or_create_collection("medical_v2")



# ── SMART CHUNKING ────────────────────────────────

def smart_chunk(text):

    """Detect and preserve Q&A pairs, fallback to paragraph chunking."""

    chunks = []



    # Try Q&A pair detection (HuggingFace medical datasets format)

    import re

    # Pattern: ### Instruction / ### Input / ### Response blocks

    qa_blocks = re.split(r'(?=###\s*Instruction|###\s*Question|Q:|Question:)', text)



    if len(qa_blocks) > 100:  # Successfully split into Q&A pairs

        print(f"   Detected Q&A format — {len(qa_blocks):,} pairs")

        for block in qa_blocks:

            block = block.strip()

            if len(block) > 80:

                chunks.append(block[:1200])  # Cap at ~300 tokens each

    else:

        # Fallback: paragraph-based chunking

        print("   Using paragraph chunking...")

        paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 80]

        for para in paragraphs:

            if len(para) <= 1200:

                chunks.append(para)

            else:

                # Split long paragraphs into 300-word chunks

                words = para.split()

                for i in range(0, len(words), 300):

                    chunk = " ".join(words[i:i+300])

                    if len(chunk) > 80:

                        chunks.append(chunk)



    return chunks



# ── INDEX DATASET ─────────────────────────────────

def index_dataset():

    print("📚 Reading dataset.txt...")



    if not os.path.exists(DATASET):

        print(f"❌ Not found: {DATASET}")

        return



    with open(DATASET, "r", encoding="utf-8", errors="ignore") as f:

        text = f.read()



    print(f"   File size: {len(text)/1024/1024:.2f} MB")



    chunks = smart_chunk(text)

    print(f"   Total chunks: {len(chunks):,}")



    # Quick sanity check — print first 3 chunks

    print("\n   SAMPLE CHUNKS:")

    for i, c in enumerate(chunks[:3]):

        print(f"   [{i}] {c[:120]}...")

    print()



    print("   Indexing into ChromaDB...")

    batch_size = 32

    for i in range(0, len(chunks), batch_size):

        batch      = chunks[i:i+batch_size]

        embeddings = embedder.encode(batch, show_progress_bar=False).tolist()

        ids        = [f"chunk_{i+j}" for j in range(len(batch))]

        try:

            collection.add(documents=batch, embeddings=embeddings, ids=ids)

        except Exception:

            pass

        if i % 500 == 0:

            print(f"   Progress: {min(i+batch_size, len(chunks))}/{len(chunks)}")



    print(f"✅ Indexed {collection.count():,} chunks")



# ── RETRIEVAL TEST ────────────────────────────────

def test_retrieval(question):

    """Print what chunks are actually being retrieved."""

    q_embed = embedder.encode([question]).tolist()

    results = collection.query(query_embeddings=q_embed, n_results=3)

    print("\n🔎 TOP RETRIEVED CHUNKS:")

    for i, doc in enumerate(results["documents"][0]):

        print(f"\n  [{i+1}] {doc[:200]}...")



# ── PIPELINE 2 QUERY ──────────────────────────────

def ask_basic_rag(question):

    print("\n" + "="*55)

    print("🔍 PIPELINE 2: BASIC RAG (ChromaDB + Gemini)")

    print("="*55)



    # Show what's being retrieved (debug)

    test_retrieval(question)



    try:

        q_embed = embedder.encode([question]).tolist()

        results = collection.query(query_embeddings=q_embed, n_results=3)

        context = "\n\n".join(results["documents"][0])



        # Keep prompt tight — don't inflate tokens

        prompt = f"""Use the medical context below to answer the question. Be concise.



CONTEXT:

{context}



QUESTION: {question}

ANSWER:"""



        start    = time.time()

        response = client.models.generate_content(

            model=MODEL_ID,

            contents=prompt

        )

        latency = round(time.time() - start, 2)



        total_tokens = response.usage_metadata.total_token_count

        cost         = (total_tokens / 1_000_000) * 0.075



        print(f"\n📝 ANSWER:\n{response.text}")

        print("\n" + "─"*55)

        print(f"📊 METRICS:")

        print(f"• Tokens : {total_tokens:,}")

        print(f"• Latency: {latency}s")

        print(f"• Cost   : ${round(cost, 6)}")

        print("="*55)



        return {

            "answer"      : response.text,

            "latency"     : latency,

            "total_tokens": total_tokens,

            "cost_usd"    : round(cost, 6),

        }



    except errors.ClientError as e:

        if "429" in str(e):

            print("❌ Rate limit. Wait 60s.")

        else:

            print(f"❌ Error: {e}")



# ── MAIN ──────────────────────────────────────────

if __name__ == "__main__":

    if collection.count() == 0:

        print("ChromaDB empty — indexing now...")

        index_dataset()

    else:

        print(f"✅ ChromaDB ready — {collection.count():,} chunks")



    while True:

        print()

        q = input("🩺 Question (or 'quit'): ").strip()

        if not q or q.lower() == "quit":

            break

        ask_basic_rag(q)
