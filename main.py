from youtube.llm import summarize_youtube, build_youtube_query_engine, get_youtube_info, qa_youtube, load_index
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from fastapi import FastAPI, Request, Response, status
from dotenv import load_dotenv
import re
import shutil
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex

def checkDateBase(llm):
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    # chroma_storage_path = "./youtube_storage"
    # chroma_collection_name = "youtube"
    # chroma_collection = chromadb.PersistentClient(path=chroma_storage_path)\
    #                             .get_or_create_collection(chroma_collection_name)
    # vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # index = VectorStoreIndex.from_vector_store(
    #     vector_store=vector_store, 
    #     embed_model=embed_model
    # )
    index = load_index(embed_model)
    print(f"list 內容...\n{list(index.docstore.docs.values())}")

    nodes = list(index.docstore.docs.values())
    print(f"list長度...\n{len(nodes)}")
    # print(chroma_collection.get_model())

    print("\r\n..... 總結影片內容 .....\r\n")
    query_engine = index.as_query_engine(llm=llm)
    res = query_engine.query("請用繁體中文總結這篇影片的內容")
    print(res)


def main():
    llm = Ollama(model="llama3.2", request_timeout=600, temperature=0.1)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

    while True:
        user_input = input("\r\n請輸入您的問題 (輸入 'End' or 'end' 離開): ")
         # 檢查是否為空字串
        if user_input == "":
            print("您未輸入任何內容，請重新輸入。")
            continue  # 繼續下一次迴圈

        if user_input == "END" or user_input == "end":
            print("程式結束")
            break
        # print("使用者輸入的是:", user_input)

        if user_input == "查看資料庫內容":
            checkDateBase(llm=llm)
        else :
            message = user_input
            urls = re.findall(r'https?://[^\s)\]]+', message)
            if urls:
                print('收到使用者提供的影片連結:', urls)
                build_youtube_query_engine(youtube_link=message, llm=llm, embed_model=embed_model)
                res = f"\r\n已收到影片連結。\r\n你有什麼問題呢?\r\n"
                print(res)
            else:
                print('持續對話中...')
                res = get_youtube_info(query=message, llm=llm, embed_model=embed_model)
                # res = qa_youtube(user_query=message, llm=llm, embed_model=embed_model)
                print(res)

if __name__ == "__main__":
    main()
