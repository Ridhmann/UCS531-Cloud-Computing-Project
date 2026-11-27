Serverless RAG Pipeline on AWS for Document Q&A
===============================================

This project is a complete, serverless Retrieval-Augmented Generation (RAG) pipeline built on AWS. It allows a user to upload large PDF documents (like research papers) and ask questions about their content.

The system is designed to be scalable, asynchronous, and cost-effective, using AWS Lambda for compute, S3 for storage, DynamoDB as a vector-capable database, and Amazon Bedrock (with the DeepSeek model) for text generation.

## 📁 Project Structure
```
.
├── docs/
│ ├── 1-AWS_Setup.md
│ ├── 2-Upload_Endpoint.md.md
│ ├── 3-Data_Ingestion.md
│ ├── 4-RAG_Query_API.md
│ └── 5-Testing_Scripts.md
│
├── pdfs/
│ └── (sample PDF papers go here)
│
├── lambda_functions/
│ ├── get_presigned_url.py
│ ├── process_document_batch.py
│ ├── handle_textract_result.py
│ ├── generate_embeddings.py
│ ├── handle_query.py
│ └── generate_answer.py
│
├── index.ipynb
└── README.md
```

### 🧩 Directory Overview

| Directory / File | Description |
|------------------|-------------|
| **docs/** | Contains setup and API documentation files for the project. |
| **pdfs/** | Stores sample PDF documents for testing uploads. |
| **lambda_functions/** | Contains all AWS Lambda functions for processing, embeddings, querying, and generating answers. |
| **index.ipynb** | Main Jupyter Notebook used for local testing or walkthroughs. |
| **README.md** | Project overview and setup guide. |

Project Architecture
--------------------

The architecture is split into two main flows: the **Data Ingestion Pipeline** and the **RAG Query API**.

### 1\. Data Ingestion Pipeline (Asynchronous)

This flow is event-driven and designed to handle large, multi-page PDFs without timing out.

1.  **Client (Jupyter)**: A user requests a secure upload URL from API Gateway.
    
2.  **API Gateway (POST /upload-url)**: Triggers a Lambda to generate an S3 presigned URL.
    
3.  **Client (Jupyter)**: Uses the presigned URL to upload the PDF directly to S3.
    
4.  **S3 (PUT Event)**: The file upload triggers the process-document-batch Lambda.
    
5.  **Lambda 1 (process-document-batch)**: Starts an asynchronous start\_document\_text\_detection job with Amazon Textract. It tells Textract to send a notification to an SNS topic when complete.
    
6.  **SNS Topic (textract-job-complete)**: Receives the "Job Succeeded" message from Textract.
    
7.  **Lambda 2 (handle-textract-result)**: Is triggered by the SNS message. It fetches the full text from Textract, splits it into chunks, and batch-writes the chunks to DynamoDB with a status of PENDING.
    
8.  **EventBridge (CRON Job)**: Every 5 minutes, this triggers the generate-embeddings Lambda.
    
9.  **Lambda 3 (generate-embeddings)**: Scans DynamoDB for all chunks with PENDING or FAILED status.
    
10.  **Amazon Bedrock (Titan)**: The Lambda calls the Titan Embeddings model to get a vector for each text chunk.
    
11.  **DynamoDB**: The Lambda updates the items with their vector embedding and sets the status to COMPLETED.
    

### 2\. RAG Query API (Synchronous)

This flow is a standard API that allows a user to ask a question and get a single, synthesized answer.

1.  **Client (Jupyter)**: Sends a JSON request ({"question": "..."}) to the /query endpoint.
    
2.  **API Gateway (POST /query)**: Triggers the handle-query Lambda.
    
3.  **Lambda 4 (handle-query)**:a. Generates an embedding for the user's _question_ (using the same Titan model).b. Scans DynamoDB for all COMPLETED chunks.c. Calculates the cosine similarity between the question's vector and every chunk's vector.d. Returns the top\_k (e.g., top 5) most similar chunks.
    
4.  **Client (Jupyter)**: Receives the top chunks and immediately calls the /answer endpoint, passing the question and the chunks.
    
5.  **API Gateway (POST /answer)**: Triggers the generate-answer Lambda.
    
6.  **Lambda 5 (generate-answer)**:a. Builds a new prompt containing the user's question and the "context" from the retrieved chunks.b. Calls **Amazon Bedrock (DeepSeek)** with this augmented prompt.
    
7.  **Client (Jupyter)**: Receives the final, LLM-generated answer and the source chunks used to create it.
    

Implementation Guide
--------------------

The full implementation is broken down into logical parts. Please see the guides in the docs/ folder for step-by-step instructions.

*   **Part 1: AWS Infrastructure Setup**
    
    *   [Guide to setting up S3, DynamoDB, and the IAM Role.](docs/1-Aws_Setup.md)
        
*   **Part 2: The Upload Endpoint**
    
    *   [Guide to creating the get\_presigned\_url Lambda and POST /upload-url API.](docs/2-Upload_Endpoint.md)
        
*   **Part 3: Asynchronous Ingestion Pipeline**
    
    *   [Guide to setting up the core data pipeline: SNS, Textract Lambdas, and the EventBridge scheduler.](docs/3-Data_Ingestion.md)
        
*   **Part 4: The RAG Query API**
    
    *   [Guide to creating the /query and /answer endpoints and their Lambda functions.](docs/4-RAG_Query_API.md)
        
*   **Part 5: Testing Scripts**
    
    *   [Guide to using the Jupyter Notebook (index.ipynb) to test the full end-to-end pipeline.](docs/5-Testing_Scripts.md)

Web Front End (Streamlit)
-------------------------

`streamlit_app.py` is a web UI for the same API Gateway endpoints the notebook uses. It does no processing locally:

*   **Upload** (sidebar): `POST /upload-url` returns a presigned URL, and the app PUTs the PDF straight to S3. Ingestion then runs asynchronously (allow 5-10 minutes).
*   **Ask**: `POST /query` retrieves the top-k chunks, then `POST /answer` generates the answer with Bedrock. The app shows the answer and the source chunks with their similarity scores.
*   You can query the five sample papers already ingested, any document uploaded in the session, a document ID you enter, or all documents.

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The API base URL defaults to the `dev` stage used in `index.ipynb`. To point at a different deployment, set `API_BASE_URL` as an environment variable or in `.streamlit/secrets.toml`, or edit it in the sidebar.

Sample Questions
-----------------

[Guide to sample questions for each doc.](docs/Sample_Questions.md)