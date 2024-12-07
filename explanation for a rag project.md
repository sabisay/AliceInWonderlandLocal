<h4> A RAG project can be generalized into key steps: embedding the language, importing datasets, and creating chunks, crafting an effective prompt, and adding a grader to evaluate results. Beyond this, incorporating memory and retrieval mechanisms, such as Chroma or FAISS, is essential for efficient data access. Evaluation metrics like BLEU or ROUGE help measure performance, while deployment considerations ensure real-world usability. Optimizing each of these steps, particularly prompt design and data chunking, is crucial for building robust and scalable RAG solutions.</h4>

for setup the llm into our project we can use these seperate steps:
<p>
Local LLM setup
local_llm = "llama3.2"
local_llm_instance = ChatOllama(model=local_llm, temperature=0)
local_llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")
</p>

# API-based LLM setup (e.g., OpenAI)
api_llm = ChatOpenAI(
    model="gpt-4",
    openai_api_key="your-api-key-here",
    temperature=0,
    max_tokens=1000  # Optional: set token limits
)

