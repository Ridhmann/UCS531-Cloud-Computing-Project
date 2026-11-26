import streamlit as st
import pymupdf
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title="Cloud PDF AI Assistant",
    page_icon="📄"
)

st.title("Cloud PDF AI Assistant")
st.write("Upload a PDF and ask questions about its content.")

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


pdf = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if pdf is not None:
    try:
        with pymupdf.open(
            stream=pdf.getvalue(),
            filetype="pdf"
        ) as doc:
            pages = [page.get_text() for page in doc]

    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        st.stop()

    st.success(f"PDF loaded: {len(pages)} pages")

    question = st.text_input(
        "Ask a question about the PDF"
    )

    if question.strip():
        words = {
            word for word in re.findall(
                r"\w+", question.lower()
            )
            if len(word) > 2
        }

        ranked = sorted(
            enumerate(pages),
            key=lambda item: sum(
                item[1].lower().count(word)
                for word in words
            ),
            reverse=True
        )

        relevant = [
            (i, page)
            for i, page in ranked
            if any(
                word in page.lower()
                for word in words
            )
        ][:2]

        if not relevant:
            st.warning("No relevant text found in the PDF.")

        elif st.button("Generate AI answer"):
            try:
                with st.spinner("Loading AI model and generating answer..."):
                    tokenizer, model = load_model()

                    context = " ".join(
                        page[:1000] for _, page in relevant
                    )

                    prompt = (
                        "Answer the question using only the context. "
                        "If the answer is not in the context, say so.\n"
                        f"Question: {question}\n"
                        f"Context: {context}\n"
                        "Answer:"
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
                            max_new_tokens=100
                        )

                    answer = tokenizer.decode(
                        output[0],
                        skip_special_tokens=True
                    )

                st.subheader("AI Answer")
                st.write(answer)

                st.caption(
                    "Source pages: " +
                    ", ".join(
                        str(i + 1) for i, _ in relevant
                    )
                )

            except Exception as e:
                st.error(f"AI model error: {e}")
