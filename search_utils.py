from sentence_transformers import SentenceTransformer, util
import torch

# model = SentenceTransformer('bert-base-nli-mean-tokens')


def search_in_corpus(model, sentence, corpus):
    """python function to search a sentence in a corpus using sentence embedding library"""
    corpus_embeddings = model.encode(corpus)
    sentence_embedding = model.encode(sentence)
    # Compute cosine-similarities for each sentence with the query sentence
    cos_scores = util.pytorch_cos_sim(sentence_embedding, corpus_embeddings)[0]
    # Sort the sentences in corpus by cosine similarity
    cos_scores = cos_scores.cpu()
    top_results = torch.topk(cos_scores, k=5)
    print("\n\n======================\n\n")
    print("Query:", sentence)
    print("\nTop 5 most similar sentences in corpus:")
    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: %.4f)" % (score))


def initialize_corpus(model, sentences):
    """python function to initialize corpus"""
    corpus_embeddings = model.encode(sentences)
    return corpus_embeddings


def embed_sentence(model, sentence):
    """python function to embed a sentence using sentence embedding library"""
    sentence_embedding = model.encode(sentence)
    return sentence_embedding

