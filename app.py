"""AI College Assistant for E&TC students.
Streamlit + Claude API + PyMuPDF + ChromaDB (RAG over your own PDF notes)."""
import base64
import hashlib
import json
import os
import re

import anthropic
import chromadb
import fitz  # PyMuPDF
import streamlit as st

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
CHUNK, OVERLAP = 1200, 200

SYSTEM = """You are a patient teaching assistant for Electronics & Telecommunication (E&TC) \
engineering students: analog/digital electronics, communication systems, signals & systems, \
control systems, MATLAB and Arduino/embedded C.
Rules:
- When NOTES CONTEXT is provided, ground your answer in it and cite sources like [manual.pdf p.12].
- If the notes don't cover something, say so, then use standard textbook knowledge, \
clearly marked "Beyond your notes".
- Never invent measurements or experimental results.
- Use LaTeX for equations ($...$ or $$...$$) and fenced blocks for code.
- Be concise, exam-oriented and at undergraduate level."""

st.set_page_config(page_title="E&TC AI Assistant", page_icon="📡", layout="wide")


# ---------- Vector store ----------
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")
    # Uses Chroma's built-in local embedding model (no extra API key needed)
    return client.get_or_create_collection("etc_notes", metadata={"hnsw:space": "cosine"})


col = get_collection()


def chunk_text(text):
    step = CHUNK - OVERLAP
    return [text[i:i + CHUNK] for i in range(0, len(text), step) if text[i:i + CHUNK].strip()]


def ingest(file):
    data = file.getvalue()
    doc_id = hashlib.md5(data).hexdigest()[:10]
    if col.get(where={"doc_id": doc_id}, limit=1)["ids"]:
        return 0, 0
    pdf = fitz.open(stream=data, filetype="pdf")
    ids, docs, metas, empty = [], [], [], 0
    for pno, page in enumerate(pdf, 1):
        text = page.get_text().strip()
        if not text:
            empty += 1
            continue
        for ci, ch in enumerate(chunk_text(text)):
            ids.append(f"{doc_id}-{pno}-{ci}")
            docs.append(ch)
            metas.append({"doc_id": doc_id, "source": file.name, "page": pno, "chunk": ci})
    for i in range(0, len(ids), 200):
        col.add(ids=ids[i:i + 200], documents=docs[i:i + 200], metadatas=metas[i:i + 200])
    return len(ids), empty


def list_sources():
    metas = col.get(include=["metadatas"])["metadatas"]
    return sorted({m["source"] for m in metas})


def retrieve(query, sources, k=8):
    if not sources:
        return []
    res = col.query(query_texts=[query], n_results=k, where={"source": {"$in": sources}})
    return [(m, d) for m, d in zip(res["metadatas"][0], res["documents"][0])]


def pages_text(sources, start, end, cap=60000):
    where = {"$and": [{"source": {"$in": sources}}, {"page": {"$gte": start}}, {"page": {"$lte": end}}]}
    res = col.get(where=where, include=["metadatas", "documents"])
    rows = sorted(zip(res["metadatas"], res["documents"]),
                  key=lambda r: (r[0]["source"], r[0]["page"], r[0]["chunk"]))
    return context_block(rows)[:cap]


def context_block(rows):
    return "\n\n".join(f"[{m['source']} p.{m['page']}]\n{d}" for m, d in rows)


# ---------- Claude ----------
def get_client():
    key = st.session_state.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        st.error("Add your Anthropic API key in the sidebar.")
        st.stop()
    return anthropic.Anthropic(api_key=key)


def stream_claude(messages, max_tokens=4000):
    with get_client().messages.stream(model=MODEL, max_tokens=max_tokens,
                                      system=SYSTEM, messages=messages) as s:
        for t in s.text_stream:
            yield t


def ask_once(prompt, max_tokens=4000):
    r = get_client().messages.create(model=MODEL, max_tokens=max_tokens, system=SYSTEM,
                                     messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if b.type == "text")


def build_prompt(task, query, sources, use_notes, extra_ctx=""):
    ctx = extra_ctx or (context_block(retrieve(query, sources)) if use_notes else "")
    return (f"NOTES CONTEXT:\n{ctx}\n\n" if ctx else "") + f"TASK:\n{task}"


def run(task, query, sources, use_notes, extra_ctx="", images=None):
    prompt = build_prompt(task, query, sources, use_notes, extra_ctx)
    content = prompt
    if images:
        content = [{"type": "image", "source": {"type": "base64", "media_type": f.type,
                                                 "data": base64.b64encode(f.getvalue()).decode()}}
                   for f in images] + [{"type": "text", "text": prompt}]
    with st.spinner("Thinking..."):
        pass
    out = st.write_stream(stream_claude([{"role": "user", "content": content}]))
    st.download_button("Download as .md", out, file_name="etc_assistant_output.md")


# ---------- Sidebar ----------
with st.sidebar:
    st.title("📡 E&TC Assistant")
    st.text_input("Anthropic API key", type="password", key="api_key")
    files = st.file_uploader("Upload PDF notes / manuals", type="pdf", accept_multiple_files=True)
    if files and st.button("Index uploaded PDFs"):
        for f in files:
            with st.spinner(f"Indexing {f.name}..."):
                n, empty = ingest(f)
            if n == 0 and empty == 0:
                st.info(f"{f.name}: already indexed.")
            else:
                st.success(f"{f.name}: {n} chunks indexed.")
            if empty:
                st.warning(f"{f.name}: {empty} page(s) had no text (scanned?). "
                           "Use Diagram & Image mode for those.")
    all_sources = list_sources()
    sources = st.multiselect("Search in", all_sources, default=all_sources)
    use_notes = st.toggle("Use my notes (RAG)", value=True)
    if all_sources and st.button("Clear all indexed notes"):
        col.delete(where={"source": {"$in": all_sources}})
        st.rerun()
    st.divider()
    mode = st.radio("Mode", ["Ask my notes", "Summarize chapter", "Viva Q&A", "Explain MATLAB",
                             "Debug Arduino", "Experiment conclusion", "MCQ quiz",
                             "Diagram & image analysis"])

st.header(mode)

# ---------- Modes ----------
if mode == "Ask my notes":
    st.session_state.setdefault("chat", [])
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if q := st.chat_input("e.g. Explain the Nyquist plot experiment and generate 15 viva questions"):
        with st.chat_message("user"):
            st.markdown(q)
        history = st.session_state.chat[-6:]
        msg = build_prompt(f"Answer the student's question.\n\n{q}", q, sources, use_notes)
        with st.chat_message("assistant"):
            ans = st.write_stream(stream_claude(history + [{"role": "user", "content": msg}]))
        st.session_state.chat += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]

elif mode == "Summarize chapter":
    by_pages = st.checkbox("Summarize by page range")
    if by_pages and sources:
        c1, c2 = st.columns(2)
        start = c1.number_input("From page", 1, 2000, 1)
        end = c2.number_input("To page", 1, 2000, 10)
        style = st.selectbox("Style", ["Exam revision notes", "Short paragraph", "Key formulas only"])
        if st.button("Summarize"):
            text = pages_text(sources, start, end)
            if not text:
                st.warning("No text found in that page range.")
            else:
                run(f"Summarize these pages as: {style}. Include key definitions, formulas and "
                    "important diagrams mentioned.", "", sources, False, extra_ctx=text)
    else:
        topic = st.text_input("Chapter / topic name")
        if topic and st.button("Summarize"):
            run(f"Summarize the chapter/topic '{topic}' as exam revision notes: definitions, "
                "key formulas, and important points.", topic, sources, use_notes)

elif mode == "Viva Q&A":
    topic = st.text_input("Experiment / topic", "Nyquist plot experiment")
    c1, c2 = st.columns(2)
    n = c1.slider("Number of questions", 5, 30, 15)
    level = c2.selectbox("Difficulty", ["Basic", "Mixed", "Tough (examiner-style)"])
    if st.button("Generate viva questions"):
        run(f"Generate {n} {level} viva questions with model answers (2-4 lines each) on '{topic}'. "
            "Base them on the student's notes where possible. Group by: concept, procedure, "
            "observation/result, application.", topic, sources, use_notes)

elif mode == "Explain MATLAB":
    code = st.text_area("Paste MATLAB code", height=250)
    if code and st.button("Explain"):
        run("Explain this MATLAB program: purpose, line-by-line walkthrough, what each plot/output "
            "shows, expected output, and 3 viva questions.\n\n```matlab\n" + code + "\n```",
            "MATLAB " + code[:300], sources, use_notes)

elif mode == "Debug Arduino":
    code = st.text_area("Paste Arduino sketch", height=250)
    info = st.text_area("Board, wiring, error message or what goes wrong", height=100)
    if code and st.button("Debug"):
        run("Debug this Arduino sketch. List the bugs (compile errors, logic, wiring/pin, timing), "
            "explain why each happens, then give the corrected full sketch.\n"
            f"Details: {info}\n\n```cpp\n{code}\n```", info or code[:300], sources, use_notes)

elif mode == "Experiment conclusion":
    title = st.text_input("Experiment title / aim")
    obs = st.text_area("Your observations / results (values, graph behaviour, errors)", height=150)
    if title and st.button("Generate conclusion"):
        run(f"Write a lab-journal conclusion for the experiment '{title}'. Student's observations:\n"
            f"{obs or '(none given)'}\nRelate results to theory, mention sources of error and "
            "one-line learning outcome. Use ONLY the observations given; if none, write a "
            "theory-based conclusion with placeholders like [value].", title, sources, use_notes)

elif mode == "MCQ quiz":
    topic = st.text_input("Topic", "Stability of control systems")
    c1, c2 = st.columns(2)
    n = c1.slider("Questions", 3, 20, 5)
    level = c2.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    if st.button("Generate quiz"):
        ctx = context_block(retrieve(topic, sources)) if use_notes else ""
        raw = ask_once(f"{'NOTES CONTEXT:' + chr(10) + ctx if ctx else ''}\n\nCreate {n} {level} MCQs on "
                       f"'{topic}'. Return ONLY a JSON array, no prose: "
                       '[{"question": str, "options": [4 strings], "answer": index 0-3, '
                       '"explanation": str}]')
        try:
            st.session_state.quiz = json.loads(re.search(r"\[.*\]", raw, re.S).group(0))
            st.session_state.checked = False
        except Exception:
            st.error("Couldn't parse the quiz. Try again.")
    quiz = st.session_state.get("quiz")
    if quiz:
        for i, q in enumerate(quiz):
            st.markdown(f"**Q{i + 1}. {q['question']}**")
            st.radio("opts", q["options"], index=None, key=f"q{i}", label_visibility="collapsed")
        if st.button("Check answers"):
            st.session_state.checked = True
        if st.session_state.get("checked"):
            score = 0
            for i, q in enumerate(quiz):
                pick = st.session_state.get(f"q{i}")
                ok = pick is not None and q["options"].index(pick) == q["answer"]
                score += ok
                st.markdown(f"{'✅' if ok else '❌'} **Q{i + 1}** - answer: {q['options'][q['answer']]}  \n"
                            f"_{q['explanation']}_")
            st.success(f"Score: {score}/{len(quiz)}")

elif mode == "Diagram & image analysis":
    imgs = st.file_uploader("Circuit / block diagram / handwritten notes",
                            type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
    ask = st.text_area("What do you want?", "Identify the components, explain how it works, "
                       "and write the transfer function / key equations.")
    if imgs:
        st.image([i.getvalue() for i in imgs], width=300)
    if imgs and st.button("Analyze"):
        run("Analyze the attached image(s) (circuit diagram, block diagram or handwritten notes). "
            "Transcribe handwriting if present and flag anything unclear instead of guessing. "
            f"Request: {ask}", ask, sources, use_notes, images=imgs)
