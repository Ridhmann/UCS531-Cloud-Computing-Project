import streamlit as st
import pymupdf

st.title("Cloud PDF Search")

pdf = st.file_uploader("Upload a PDF", type=["pdf"])

if pdf:
    with pymupdf.open(
        stream=pdf.getvalue(), filetype="pdf"
    ) as doc:
        pages = [page.get_text() for page in doc]

    query = st.text_input("Search text")

    if query:
        found = False
        for i, page in enumerate(pages):
            if query.lower() in page.lower():
                st.write(f"Page {i + 1}")
                st.write(page)
                found = True

        if not found:
            st.warning("No match found")
