<h4> A RAG project can be generalized into key steps: embedding the language, importing datasets, and creating chunks, crafting an effective prompt, and adding a grader to evaluate results. Beyond this, incorporating memory and retrieval mechanisms, such as Chroma or FAISS, is essential for efficient data access. Evaluation metrics like BLEU or ROUGE help measure performance, while deployment considerations ensure real-world usability. Optimizing each of these steps, particularly prompt design and data chunking, is crucial for building robust and scalable RAG solutions.</h4></br>

<h5>for setup the llm into our project we can use these seperate steps:</h5>
<pre>
    //Local LLM setup
    local_llm = "llama3.2"
    local_llm_instance = ChatOllama(model=local_llm, temperature=0)
    local_llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")
</pre>

<pre>
    //API-based LLM setup (e.g., OpenAI)
    api_llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key="your-api-key-here",
    temperature=0,
    max_tokens=1000  # Optional: set token limits
)
</pre>

<h5>loading and splitting the documents:</h5>
<pre>
    with open(r'yourFilePath.md', 'r', encoding='utf-8') as file: 
    document = file.read()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents([document])
    print(f"Number of chunks created: {len(doc_splits)}")
</pre>

<h5>setting up embeddings and creating chunks:</h5>
<pre>
    //Load the document from the .md file
    with open(r'yourFilePath.md', 'r', encoding='utf-8') as file: 
    document = file.read()
    //Split the document into chunks
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=200
    )
    doc_splits = text_splitter.split_documents([document])
    // Local Embedding Setup
    local_embedding = LocalEmbedding(model="nomic-embed-text-v1.5", inference_mode="local")
    vectorstore_local = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=local_embedding,
    )
    // API Embedding Setup (e.g., OpenAI)
    openai_embedding = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore_openai = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=openai_embedding,
    )
</pre>
