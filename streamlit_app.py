import streamlit as st
import pymupdf

st.title("Cloud PDF Question Answering")

pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf:
    with pymupdf.open(
        stream=pdf.getvalue(), filetype="pdf"
    ) as doc:
        pages = [page.get_text() for page in doc]

    question = st.text_input("Ask a question about the PDF")

    if question:
        words = [
            w.lower() for w in question.split()
            if len(w) > 2
        ]

        results = []
        for i, page in enumerate(pages):
            score = sum(
                page.lower().count(w) for w in words
            )
            if score > 0:
                results.append((score, i, page))

        results.sort(key=lambda x: x[0], reverse=True)

        if results:
            st.subheader("Relevant PDF passages")
            for score, i, page in results[:3]:
                st.write(f"Page {i + 1}")
                st.write(page[:1500])
        else:
            st.warning("No relevant text found.")
