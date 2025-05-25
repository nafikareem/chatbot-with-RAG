import os
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain.chains import RetrievalQA
import time
import torch

# sample text for testing
sample_text = """\
This is a sample text for testing the LangChain pipeline.
It contains multiple sentences and paragraphs to ensure proper text splitting and embedding.
The goal is to create a vector store and perform retrieval-based question answering.
"""

def test_rag_pipeline():
    print('starting RAG pipeline test')
    
    # step 1: simulation document loading
    loader = TextLoader(os.path.join('data', 'sample.txt'), encoding='utf-8')
    documents = loader.load()
    print(f'Documents loaded: {len(documents)}')
    print(f'Document content: {documents[0].page_content if documents else 'No content'}')
    print(f'Document type: {type(documents[0]) if documents else 'No document'}')
    
    # step 2: text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=5)
    chunks = text_splitter.split_documents(documents)
    print(f'Chunks created: {len(chunks)}')

    if not chunks:
        print("No chunks were created. Please check your chunk_size or input document.")
        return
    
    # step 3: embedding
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    start = time.time()
    vector_store = FAISS.from_documents(chunks, embeddings)
    end = time.time()
    print(f'Embedding completed in {end - start:.2f} seconds')
    print(f'Vector store created with {vector_store.index.ntotal} vectors')
    
    # step 4: reranking (take the top-k documents)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    relevant_docs = retriever.get_relevant_documents("What is the purpose of this text?")
    print(f'Retrieved {len(relevant_docs)} relevant documents')
    
    # step 5: question answering
    hf_pipeline = pipeline('text2text-generation', model='google/flan-t5-base', device=0 if torch.cuda.is_available() else -1)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="Answer the question based on the context: {context}\nQuestion: {question}"
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    #step 6: test querry
    querry = "What is the purpose of this text?"
    start = time.time()
    result = qa_chain({"query": querry})
    end = time.time()
    print(result)
    print(f'Generated answer: {result.get("answer", result.get("result", "No answer found"))}')
    print(f'Time taken for question answering: {end - start:.2f} seconds')
    
    # simple verification
    assert result.get('result', '').strip() != '', "The answer is empty."
    print('RAG pipeline test completed successfully')
    
if __name__ == '__main__':
    # save sample text to file for testing
    if not os.path.exists('data'):
        os.makedirs('data')
    with open(os.path.join('data', 'sample.txt'), 'w', encoding='utf-8') as f:
        f.write(sample_text)

    test_rag_pipeline()
    print('RAG pipeline test completed')