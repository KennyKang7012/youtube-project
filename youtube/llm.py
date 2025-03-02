from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.core import VectorStoreIndex, SummaryIndex, StorageContext
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import ChatPromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb, os, re

chroma_storage_path = "./youtube_storage"
chroma_collection_name = "youtube"

TEXT_QA_SYSTEM_PROMPT = ChatMessage(
    content=(
        "You are an expert Q&A system that is trusted around the world.\n"
        "Always answer the query using the provided context information, "
        "and not prior knowledge.\n"
        "Some rules to follow:\n"
        "1. Never directly reference the given context in your answer.\n"
        "2. Avoid statements like 'Based on the context, ...' or "
        "'The context information ...' or anything along "
        "those lines.\n"
        "3. Detect the language of the question, and always answer in that language"
    ),
    role=MessageRole.SYSTEM,
)

TEXT_QA_PROMPT_TMPL_MSGS = [
    TEXT_QA_SYSTEM_PROMPT,
    ChatMessage(
        content=(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query.\n"
            "Query: {query_str}\n"
            "Answer: "
        ),
        role=MessageRole.USER,
    ),
]

MY_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)

def load_data(youtube_link):
    print(f"影片連結 : '{youtube_link}'")
    loader = YoutubeTranscriptReader()
    documents = loader.load_data(
        ytlinks=[youtube_link],
        languages=["zh", "zh-TW", "en"] # 優先度依序遞減
    )

    if len(documents) == 0:
        return f"Oops...沒有找到任何與'{youtube_link}'有關的影片連結"
    return documents

def build_index(documents, embed_model):
    db = chromadb.PersistentClient(path=chroma_storage_path)
    chroma_collection = db.get_or_create_collection(chroma_collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 設定 storage_context 成儲存向量至本地 SQLite 資料庫
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 文件向量化
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        embed_model=embed_model
    )
    return index

def load_index(embed_model):
    db = chromadb.PersistentClient(path=chroma_storage_path)
    chroma_collection = db.get_or_create_collection(chroma_collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )
    return index

def get_youtube_info(query, llm, embed_model):
    index = load_index(embed_model)

    # Customize my own prompt
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=2,
        # text_qa_template=MY_QA_PROMPT,
    )

    # Another way to set the prompt of a query engine:
    query_engine.update_prompts({
       "response_synthesizer:text_qa_template": MY_QA_PROMPT
    })

    res = query_engine.query(query)
    # print(res)
    return res.response

def qa_youtube(user_query, llm, embed_model):

    index = load_index(embed_model)

    query_engine = index.as_query_engine(llm=llm)

    system_query = ""

   # M (任務目標)
    system_query += "若提問與YouTube影片內容解析無關，或是資料來源無法找到相關的資訊，請以禮貌的方式拒絕，並清楚告知你的服務範疇\n"

    # A (指派對象)
    system_query += "你是一位熟悉影音內容及網路趨勢的專業影片分析師\n"

    # T (任務說明)
    system_query += "請負責從提供的YouTube影片描述的內容中，回答使用者所詢問的任何問題\n"

    # E (期望成果)
    system_query += "開頭請先以輕鬆俏皮的語氣簡短介紹你的身份，請全部用繁體中文回應，請勿使用任何簡體中文，除非專有名詞，否則請勿中英文夾雜，不使用任何Markdown語法回應\n"

    system_query += f"底下是使用者的提問: {user_query}"

    response = query_engine.query(system_query)
    print(response)
    return response.response

def build_youtube_query_engine(youtube_link, llm, embed_model):
    if not llm or not embed_model:
        raise Exception("No llm or embed_model")
    
    urls = re.findall(r'https?://[^\s)\]]+', youtube_link)
    if not os.path.exists(chroma_storage_path):
        print("Empty Data Base...")
        # Load Data
        documents = load_data(youtube_link)
        
        # Build Indexing
        index = build_index(documents, embed_model)
    elif urls:
        print("Add New urls...")
        # Load Data
        documents = load_data(youtube_link)
        
        # Build Indexing
        index = build_index(documents, embed_model)
    else:
        print("Load Indexing...")
        # Load Indexing
        index = load_index(embed_model)
    return index

def summarize_youtube(youtube_link, llm, embed_model):
    if not llm or not embed_model:
        raise Exception("No llm or embed_model")
    
    print(f"影片連結 : '{youtube_link}'")

    loader = YoutubeTranscriptReader()
    documents = loader.load_data(
        ytlinks=[youtube_link],
        languages=["zh", "zh-TW", "en"] # 優先度依序遞減
    )

    if len(documents) == 0:
        return f"Oops...沒有找到任何與'{youtube_link}'有關的影片連結"

    index = VectorStoreIndex.from_documents(
        documents=documents, 
        embed_model=embed_model
    )

    query_engine = index.as_query_engine(llm=llm, similarity_top_k=2)
    res = query_engine.query("請用繁體中文總結這篇文章的內容，並條列式列出所有重點")
    # print(res)
    return res.response

# llm = Ollama(model="llama3.2", request_timeout=600)
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
# summarize = summarize_youtube("https://www.youtube.com/watch?v=PuUshL-sWco", llm, embed_model)
# print(summarize)