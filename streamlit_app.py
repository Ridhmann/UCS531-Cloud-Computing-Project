import streamlit as st
import pymupdf
import re
from transformers import pipeline

st.title("Cloud PDF AI Assistant")

pdf = st.file_uploader("Upload a PDF", type=["pdf"])

@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1
    )

if pdf:
    with pymupdf.open(
        stream=pdf.getvalue(), filetype="pdf"
    ) as doc:
        pages = [page.get_text() for page in doc]

    question = st.text_input("Ask a question")

    if question:
        words = set(re.findall(r"\w+", question.lower()))
        words = {w for w in words if len(w) > 2}

        ranked = sorted(
            enumerate(pages),
            key=lambda item: sum(
                item[1].lower().count(w) for w in words
            ),
            reverse=True
        )

        relevant = [
            (i, page) for i, page in ranked
            if any(w in page.lower() for w in words)
        ][:2]

        if not relevant:
            st.warning("No relevant text found.")
        else:
            context = " ".join(
                page[:1200] for _, page in relevant
            )

            if st.button("Generate AI answer"):
                with st.spinner("Generating answer..."):
                    model = load_model()
                    prompt = (
                        "Answer using only this context. "
                        "If the answer is not present, say so.\n"
                        f"Context: {context}\n"
                        f"Question: {question}\nAnswer:"
                    )
                    result = model(
                        prompt,
                        max_new_tokens=100,
                        truncation=True
                    )
                    st.write(result[0]["generated_text"])

                st.caption(
                    "Source pages: " +
                    ", ".join(str(i + 1) for i, _ in relevant)
                )
