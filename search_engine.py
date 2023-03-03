import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from search_utils import initialize_corpus


class SearchEngine:

    def __init__(self):
        corpus = ['A man is eating food.',
                  'A man is eating a piece of bread.',
                  'The girl is carrying a baby.',
                  'A man is riding a horse.',
                  'A woman is playing violin.',
                  'Two men pushed carts through the woods.',
                  'A man is riding a white horse on an enclosed ground.',
                  'A monkey is playing drums.',
                  'A cheetah is running behind its prey.'
                  ]
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.corpus_embeddings = initialize_corpus(self.model, corpus)

    def search_in_corpus_embeddings(self, queries):
        # Compute embedding for each query
        query_embeddings = self.model.encode(queries)

        # Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity

        top_k = min(5, len(queries))
        for query in queries:
            query_embedding = self.model.encode(query, convert_to_tensor=True)

            # We use cosine-similarity and torch.topk to find the highest 5 scores
            cos_scores = cos_sim(query_embedding, self.corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            print("\n\n======================\n\n")
            print("Query:", query)
            print("\nTop 5 most similar sentences in corpus:")

            for score, idx in zip(top_results[0], top_results[1]):
                print(top_results[idx], "(Score: {:.4f})".format(score))
