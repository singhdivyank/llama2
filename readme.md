# Using Llama2 as an alternate to ChatGPT for question answering over a file

Meta released open source [Llama2](https://ai.meta.com/llama/) as an LLM model, this repository attempts to utilise it for question answering over a pdf/word/csv file instead of using ChatGPT API

There are two files- `download_version.py` and `using_llama.py`

* In the former a quantized Llama2 model is downnloaded and implemented using CTransformer and LangChain
* In the later Llama2 model from Meta repository and implemented as HuggingFaceLLM using Llama_index framework

## Flow
1. Upload document
2. Create embeddings
3. Store in vector store (FAISS/ SimpleDirectoryReader)
4. User query
5. Create embeddings
6. Fetch documents
7. Send this to Llama

## Requirements
1. Langchain (designing prompts, history retention and chains)
2. llama_index (for HuggingFace implementation)
3. FAISS (vector store)
4. Transformer for embeddings (HuggingFace `Sentence Transformer`)
5. Llama2 (serves as LLM)
6. Gradio (UI)

## Instructions

`pip install -r requirements.txt`

**For using a downloaded llama2 quantized model (downloaded_version.py)**: 
1. Download a Llama2 model from- https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
2. update model code in line 29: `self.llama_model = "<'downloaded_model_name'>"`
3. `python3 download_version.py`
4. Upload a .pdf/.csv/.docx file and type in your question on the Gradio UI

Note: there are many available models, you can choose any as per the `model card`

**For using Llama as HuggingFaceLLm (using_llama.py)**:
1. get access to `meta-llama`
2. create a HuggingFace access token to access the repo
3. shortlist any llama2 model offered, and update line 28: `self.model_name = "<'llama2_model_name'>"`
4. `export HF_KEY = <'huggingface_access_token'>`
5. `python3 using_llama.py`
6. Upload a .pdf/.csv/.docx file and type in your question on the Gradio UI

Note: currently implemented model, `Llama-2-7b-chat-hf`

**A word of caution: this iplementation consumes time and is memory expensive**

## Sample UI
![](image.png)