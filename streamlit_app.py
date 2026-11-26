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
                    tokenizer, model = load_model()

                    stopwords = {
                        "what", "which", "when", "where",
                        "explain", "describe", "about",
                        "does", "with", "from", "the",
                        "and", "are", "for", "how", "why",
                        "this", "that", "give", "example"
                    }

                    words = {
                        w for w in re.findall(
                            r"\w+", question.lower()
                        )
                        if len(w) > 2 and w not in stopwords
                    }

                    chunks = []

                    for page_num, page in enumerate(pages):
                        page_words = page.split()

                        for start in range(0, len(page_words), 100):
                            chunk = " ".join(
                                page_words[start:start + 130]
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
                                chunks.append(
                                    (score, page_num, chunk)
                                )

                    chunks.sort(
                        key=lambda item: item[0],
                        reverse=True
                    )

                    if not chunks:
                        st.warning("No relevant text found in the PDF.")

                    else:
                        relevant = chunks[0]
                        context = relevant[2]

                        prompt = (
                            "Answer the question in complete sentences "
                            "using only the context. Include definitions "
                            "and examples when available. Do not return "
                            "only a heading. If the answer is missing, "
                            "say it is not present in the PDF.\n"
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
                                max_new_tokens=180,
                                num_beams=4,
                                repetition_penalty=1.2,
                                early_stopping=True
                            )

                        answer = tokenizer.decode(
                            output[0],
                            skip_special_tokens=True
                        )

                        st.subheader("AI Answer")
                        st.write(answer)

                        st.caption(
                            f"Source page: {relevant[1] + 1}"
                        )

                        with st.expander("View retrieved PDF text"):
                            st.write(context)

            except Exception as e:
                st.error(f"AI model error: {e}")
