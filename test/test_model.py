import torch
from transformers import pipeline, AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import time

# testing embedding model
def test_embedding_model():
    print('testing embdedding model: paraphrase-multilingual-MiniLM-L12-v2')
    try:
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
        
        text = ['halo, ini adalah tes untuk embedding model', 'hello, this is a test for embedding model']
        
        start = time.time()
        embeddings = model.encode(text)
        end = time.time()
        
        print(f'Embedding model loaded successfully')
        print(f'time taken: {end - start} seconds')
        print(f'Embeddings shape: {embeddings.shape}')
        print(f'sample embedding (first 5 values): {embeddings[0][:5]}')
    except Exception as e:
        print(f'Error testing embedding model: {e}')
        
# testing the text generation model
def test_generation_model():
    print('\n testing generation model:')
    try:
        generator = pipeline('text2text-generation', 
                            model='google/flan-t5-base', 
                            device=0 if torch.cuda.is_available() else -1)
        
        input_text = "translate English to French: The house is wonderful."
        
        start = time.time()
        result = generator(input_text, max_new_tokens=50, num_return_sequences=1)
        end = time.time()
        print(f'Generation model loaded successfully')
        print(f'time taken: {end - start} seconds')
        print(f'Generated output: {result[0]["generated_text"]}')
    except Exception as e:
        print(f'Error testing generation model: {e}')

if __name__ == '__main__':
    print('startting model tests')
    test_embedding_model()
    test_generation_model()
    print('model tests completed')