from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline, AutoTokenizer
import os
import demonsaint


os.environ["PINECONE_API_KEY"] = demonsaint.riot.pinekey


print("PDF exists:", os.path.exists("erect.pdf"))
loader = PyPDFLoader("erect.pdf")
documents = loader.load()


def preprocess_text(text):
    """Clean and normalize text content"""
    text = text.replace("\n", " ").replace("  ", " ")  # Clean whitespace
    text = text.replace("OOAD", "Object-Oriented Analysis and Design")
    text = text.replace("UML", "Unified Modeling Language")
    return text.strip()


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,  
    chunk_overlap=50,
    length_function=lambda x: len(tokenizer.encode(x)),
    separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
)


processed_chunks = text_splitter.split_documents([
    doc.__class__(page_content=preprocess_text(doc.page_content), metadata=doc.metadata)
    for doc in documents
])
print(f"Number of chunks created: {len(processed_chunks)}")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'},  
    encode_kwargs={'normalize_embeddings': True}
)

docsearch = PineconeVectorStore.from_documents(
    documents=processed_chunks,
    index_name="fire",
    embedding=embeddings
)


retriever = docsearch.as_retriever(
    search_kwargs={
        "k": 5,  
        "score_threshold": 0.65 
    },
    search_type="mmr" 
)


model = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    device=-1,  
    max_new_tokens=150,  
    truncation=True  
)


template = """Context: {context}

Question: {input}

Answer concisely using ONLY the context. If unsure, say "This isn't covered in my materials". Answer:"""
prompt = PromptTemplate.from_template(template)


def format_docs(docs):
    """Process and clean retrieved documents"""
    if not docs:
        return "No relevant information found."
    
    seen = set()
    unique_content = []
    for doc in docs:
        content = doc.page_content
        
        content = content.replace("n ", "\n").replace(" n", "\n").strip()
        if content not in seen:
            seen.add(content)
            unique_content.append(content[:300])  # Truncate long chunks
    
    return "\n\n".join(unique_content)


def validate_response(response, query):
    """Ensure response quality"""
    response = response.replace("n ", "\n").strip()
    
    invalid_triggers = {
        "person": "I focus on technical concepts, not individuals",
        "human": "I focus on technical concepts, not individuals",
        "how are you": "I'm an AI assistant here to help with technical questions"
    }
    
    for trigger, reply in invalid_triggers.items():
        if trigger in response.lower():
            return reply
    
    return response or "I couldn't find a clear answer in the materials."


chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | (lambda x: model(str(x), truncation=True)[0]["generated_text"])
)


print("Chatbot ready! Type 'exit' to quit")
while True:
    try:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break
            
        response = chain.invoke(user_input)
        final_response = validate_response(response, user_input)
        print(f"\nBot: {final_response}")
        
    except Exception as e:
        print(f"Bot: Sorry, I encountered an error. ({str(e)})")
#by shivadeepak
