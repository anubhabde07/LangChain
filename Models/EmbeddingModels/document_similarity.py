from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Sachin Tendulkar is an Indian cricket legend known as the 'Master Blaster' and the highest run-scorer in international cricket.",
    
    "Virat Kohli is an Indian cricketer famous for his aggressive batting style, consistency, and leadership across all formats.",
    
    "MS Dhoni is a former Indian captain known for his calm demeanor, finishing skills, and leading India to multiple ICC trophies.",
    
    "Rohit Sharma is an Indian opener known for his elegant batting and record-breaking double centuries in One Day Internationals.",
    
    "Brian Lara is a West Indies batting legend who holds the record for the highest individual score in Test cricket (400 not out).",
    
    "Ricky Ponting is one of Australia's greatest captains and batsmen, leading his team to multiple World Cup victories.",
    
    "Shane Warne was an Australian spin bowler regarded as one of the greatest, famous for his 'Ball of the Century'.",
    
    "Jacques Kallis is a South African all-rounder known for scoring over 10,000 runs and taking many wickets in international cricket.",
    
    "Ben Stokes is an English all-rounder known for match-winning performances, including his heroics in the 2019 World Cup final."
]


query = "tell me about virat kohli"

doc_embeddings = embedding.embed_documents(documents)
doc_embeddings = np.asarray(doc_embeddings, dtype=float)

query_embedding = embedding.embed_query(query)
query_embedding = np.asarray(query_embedding, dtype=float).reshape(1, -1)

scores = cosine_similarity(query_embedding, doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x:x[1])[-1]

print("Index : ", index)
print("Similarity score : ", score)