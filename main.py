# 1 Import necessary dependencies
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain import hub
import phoenix as px
from phoenix.trace.langchain import LangChainInstrumentor, OpenInferenceTracer
from dotenv import load_dotenv
load_dotenv()


# 2 Launch Pheonix
px.launch_app()

# 3 Open AI configuration
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_TEMPERATURE = .5

# 4 Build Langchain Application

# Document Loader
loader = DirectoryLoader('docs/', glob="**/*.*")
docs = loader.load()

# Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Build Vectorstore
vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# Pull prompt from Langchain Prompt Hub
prompt = hub.pull("rlm/rag-prompt")

# Define LLM
llm = ChatOpenAI(
    model_name=OPENAI_MODEL, temperature=0.5)


# Format Documents Function
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


#  Define the chain to run queries on
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Instantiate an OpenInferenceTracer to store your data in OpenInference format.
# OpenInference is an open standard for capturing and storing LLM application traces
# and enables production LLMapp servers to seamlessly integrate with LLM observability solutions such as Phoenix
tracer = OpenInferenceTracer()
LangChainInstrumentor(tracer=tracer).instrument()

# Prompt the user for input
while True:
    user_query = input("Enter your query (or type 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break

    # Process the user input query
    response = rag_chain.invoke(user_query)

    print(response)
