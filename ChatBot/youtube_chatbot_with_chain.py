from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Step 1a --> Indexing (Document Ingestion)
video_id = "7ARBJQn6QkM"     # ID not url
try:
	transcript_obj = YouTubeTranscriptApi()
	fetched_transcript = transcript_obj.fetch(video_id=video_id, languages=['en'])
	transcript_list = [snippet.text for snippet in fetched_transcript.snippets]
	transcript = " ".join(transcript_list)
except TranscriptsDisabled:
	print("Transcripts are disabled for this video")
	transcript=""

# Step 1b --> Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(
	chunk_size=700,
	chunk_overlap=170,
)

chunks = splitter.create_documents([transcript])

# Step 1c & 1d --> (Embedding Generation and storing at vector store)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(
	documents=chunks,
	embedding=embedding
)

# Step 2 -- > Retriever
retriever = vector_store.as_retriever(
	search_type='similarity',
	search_kwargs={"k":4}
)

# Step 3 --> Augmentation
prompt = PromptTemplate(
	template="""
	You are a helpful AI assiatant.
	Answer only from the provided transcript context.
	If the context is not sufficient, just say you don't know.

	{context}
	Question: {question}
""",
	input_variables=['context', 'question']
)

query = "How does NVIDIA make big bets on specific chips (transformers)?"

# content_text = '\n\n'.join(doc.page_content for doc in received_docs)
def format_docs(received_docs):
	content_text = '\n\n'.join(doc.page_content for doc in received_docs)
	return content_text

parallel_chain = RunnableParallel({
	'context': retriever | RunnableLambda(format_docs),
	'question': RunnablePassthrough()
})

# Step 4 --> Generation
load_dotenv()
llm = HuggingFaceEndpoint(
	model="Qwen/Qwen2.5-72B-Instruct",
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model | parser

answer = main_chain.invoke(query)
print(answer)