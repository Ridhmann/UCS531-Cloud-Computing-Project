import streamlit as st
import pymupdf
import re
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

st.title("Cloud PDF AI Assistant")

pdf = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

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

if pdf:
    with pymupdf.open(
        stream=pdf.getvalue(),
        filetype="pdf"
    ) as doc:
        pages = [page.get_text() for page in doc]

    st.success(f"PDF loaded: {len(pages)} pages")

    question = st.text_input(
        "Ask a question"
    )

    if st.button("Generate AI answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Generating answer..."):
                    tokenizer, model = load_model()

                    # Extract meaningful question words
                    words = set(
                        re.findall(
                            r"\w+",
                            question.lower()
                        )
                    )
                    stopwords = {
                        "what", "which", "when", "where",
                        "explain", "describe", "about",
                        "does", "with", "from", "the",
                        "and", "are", "for", "how"
                    }
                    words = {
                        w for w in words
                        if len(w) > 2 and w not in stopwords
                    }

                    # Split pages into smaller chunks
                    chunks = []

                    for page_num, page in enumerate(pages):
                        page_words = page.split()

                        for start in range(
                            0, len(page_words), 120
                        ):
                            chunk = " ".join(
                                page_words[start:start + 160]
                            )

                            if not chunk.strip():
                                continue

                            score = sum(
                                chunk.lower().count(w)
                                for w in words
                            )

                            if score > 0:
                                chunks.append(
                                    (score, page_num, chunk)
                                )

                    chunks.sort(
                        key=lambda x: x[0],
                        reverse=True
                    )

                    if not chunks:
                        st.warning(
                            "No relevant text found."
                        )
                    else:
                        relevant = chunks[:3]

                        context = "\n".join(
                            chunk
                            for _, _, chunk in relevant
                        )

                        prompt = (
                            "Explain the answer in complete "
                            "sentences using only the context. "
                            "Do not return only a heading. "
                            "If the answer is missing, say so.\n"
                            f"Question: {question}\n"
                            f"Context: {context}\nAnswer:"
                        )

                        inputs = tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=512
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
                            page_num + 1
                            for _, page_num, _ in relevant
                        })

                        st.caption(
                            "Source pages: " +
                            ", ".join(
                                str(p) for p in source_pages
                            )
                        )

                        with st.expander(
                            "View retrieved PDF text"
                        ):
                            st.write(context)

            except Exception as e:
                st.error(f"AI model error: {e}")
