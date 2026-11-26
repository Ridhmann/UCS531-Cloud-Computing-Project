import re
import streamlit as st
import pymupdf
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="Cloud PDF AI Assistant",
    page_icon="📄"
)

st.title("Cloud PDF AI Assistant")
st.write("Upload a PDF and ask questions about its content.")

pdf = st.file_uploader("Upload a PDF", type=["pdf"])


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-small"
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-small"
    )
    model.eval()
    return tokenizer, model


if pdf is not None:
    try:
        with pymupdf.open(
            stream=pdf.getvalue(),
            filetype="pdf"
        ) as doc:
            pages = [page.get_text() for page in doc]

        st.success(f"PDF loaded: {len(pages)} pages")

    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        st.stop()

    question = st.text_input("Ask a question")

    if st.button("Generate AI answer"):
        if not question.strip():
            st.warning("Please enter a question.")

        else:
            try:
                with st.spinner("Searching PDF and generating answer..."):
                    stopwords = {
                        "what", "which", "when", "where", "explain",
                        "describe", "about", "does", "with", "from",
                        "the", "and", "are", "for", "how", "why",
                        "this", "that", "give", "example", "examples",
                        "using", "please", "tell", "define"
                    }

                    words = {
                        w for w in re.findall(r"\w+", question.lower())
                        if len(w) > 2 and w not in stopwords
                    }

                    if not words:
                        st.warning("Please include a topic in your question.")
                        st.stop()

                    chunks = []

                    for page_num, page in enumerate(pages):
                        page_words = page.split()

                        for start in range(0, len(page_words), 80):
                            chunk = " ".join(
                                page_words[start:start + 100]
                            )

                            if not chunk.strip():
                                continue

                            chunk_words = re.findall(
                                r"\w+", chunk.lower()
                            )

                            score = sum(
                                chunk_words.count(w) for w in words
                            )

                            if score > 0:
                                chunks.append((score, page_num, chunk))

                    chunks.sort(
                        key=lambda item: item[0],
                        reverse=True
                    )

                    if not chunks:
                        st.warning("No relevant text found in the PDF.")
                        st.stop()

                    tokenizer, model = load_model()

                    instruction = (
                        "Answer using only the context. "
                        "Give a complete explanation and an example "
                        "if present. Do not return only a heading.\n"
                        f"Question: {question}\n"
                        "Context:\n"
                    )

                    suffix = "\nAnswer:"
                    max_input = 512

                    reserved = len(
                        tokenizer.encode(
                            instruction + suffix,
                            add_special_tokens=True
                        )
                    )

                    budget = max_input - reserved

                    if budget < 50:
                        st.warning("Please shorten your question.")
                        st.stop()

                    selected = []
                    context_parts = []

                    for score, page_num, chunk in chunks:
                        remaining = budget - sum(
                            len(tokenizer.encode(part, add_special_tokens=False))
                            for part in context_parts
                        )

                        if remaining <= 0 or len(selected) >= 3:
                            break

                        token_ids = tokenizer.encode(
                            chunk,
                            add_special_tokens=False
                        )[:remaining]

                        if not token_ids:
                            continue

                        part = tokenizer.decode(
                            token_ids,
                            skip_special_tokens=True
                        )

                        context_parts.append(part)
                        selected.append((page_num, part))

                    context = "\n".join(context_parts)

                    prompt = instruction + context + suffix

                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_input
                    )

                    with torch.inference_mode():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=180,
                            num_beams=4,
                            early_stopping=True
                        )

                    answer = tokenizer.decode(
                        output[0],
                        skip_special_tokens=True
                    )

                st.subheader("AI Answer")
                st.write(answer)

                source_pages = sorted({
                    page_num + 1 for page_num, _ in selected
                })

                st.caption(
                    "Source pages: " +
                    ", ".join(str(p) for p in source_pages)
                )

                with st.expander("View retrieved PDF text"):
                    for page_num, part in selected:
                        st.markdown(f"**Page {page_num + 1}**")
                        st.write(part)

            except Exception as e:
                st.error(f"AI model error: {e}")
