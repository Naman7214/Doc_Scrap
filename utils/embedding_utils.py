import types

from fastembed import TextEmbedding
from pinecone_text.sparse import BM25Encoder

model = TextEmbedding("BAAI/bge-base-en-v1.5")
bm25 = BM25Encoder().default()


def get_sparse_embedding(text):
    doc_sparse_vector = bm25.encode_documents(text)
    return {
        "indices": doc_sparse_vector["indices"],
        "values": doc_sparse_vector["values"],
    }


def get_embedding(text):
    global request_count
    request_count += 1
    print(f"Embedding text. Request count: {request_count}")
    try:
        # Get the raw embedding output, and force conversion from generator to list.
        raw = model.embed(text, batch_size=24, parallel=True)
        # If raw is a generator, exhaust it.
        if isinstance(raw, types.GeneratorType):
            raw = list(raw)

        # Now, if raw is a numpy array, convert it to a list.
        if hasattr(raw, "tolist"):
            embeddings = raw.tolist()
        # Otherwise, if it's already a list, check each element.
        elif isinstance(raw, list):
            embeddings = [
                e.tolist() if hasattr(e, "tolist") else e for e in raw
            ]
        else:
            embeddings = raw

        print("Received response")
        return embeddings[0]
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
