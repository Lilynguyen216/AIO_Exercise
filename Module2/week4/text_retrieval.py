#Download data
!gdown 1jh2p2DlaWsDo_vEWIcTrNh3mUuXd-cw6

#import lib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

vi_data_df = pd.read_csv('./vi_text_retrieval.csv')
context = vi_data_df['text']
context = [doc.lower() for doc in context]

tfidf_vectorizer = TfidfVectorizer()
context_embedded = tfidf_vectorizer.fit_transform(context)
res = round(context_embedded.toarray()[7][0], 2)
print(res)

def tfidf_search(question, tfidf_vectorizer, top_d=5):
  query_embedded = tfidf_vectorizer.transform([question])
  cosine_scores = cosine_similarity(query_embedded, context_embedded).flatten()

  results = []
  for idx in cosine_scores.argsort()[-top_d:][::-1]:
    doc_score = {
        'id': idx,
        'cosine_score': cosine_scores[idx]
    }
    results.append(doc_score)

  return results

def corr_search(question, tfidf_vectorizer, top_d=5):
  query_embedded = tfidf_vectorizer.transform([question])
  corr_scores = np.corrcoef(query_embedded.toarray(), context_embedded.toarray())
  corr_scores = corr_scores[0][1:]

  results = []
  for idx in corr_scores.argsort()[-top_d:][::-1]:
    doc = {
        'id': idx,
        'corr_score': corr_scores[idx]
    }
    results.append(doc)

  return results

question = vi_data_df.iloc[0]['question']

result1 = tfidf_search(question, tfidf_vectorizer, top_d=5)
print(round(result1[0]['cosine_score'], 2))

result2 = corr_search(question, tfidf_vectorizer, top_d=5)
print(round(result2[1]['corr_score'], 2))
