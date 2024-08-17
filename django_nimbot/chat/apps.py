from django.apps import AppConfig
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import os
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter


class ChatConfig(AppConfig):
    name = 'chat'
    current_working_directory = os.getcwd()

    print("Current working directory:", current_working_directory)

    def ready(self):
        embedder = NVIDIAEmbeddings(model="ai-embed-qa-4")

        ps = os.listdir("./chat/zh_data/")
        data = []
        sources = []
        for p in ps:
            if p.endswith('.txt'):
                path2file = "./chat/zh_data/" + p
                with open(path2file, encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        if len(line) >= 1:
                            data.append(line)
                            sources.append(path2file)

        documents = [d for d in data if d != '\n']
        text_splitter = CharacterTextSplitter(chunk_size=400, separator=" ")
        docs = []
        metadatas = []

        for i, d in enumerate(documents):
            splits = text_splitter.split_text(d)
            docs.extend(splits)
            metadatas.extend([{"source": sources[i]}] * len(splits))

        store = FAISS.from_texts(docs, embedder, metadatas=metadatas)
        store.save_local('./chat/zh_data/nv_embedding')
        store = FAISS.load_local("./chat/zh_data/nv_embedding", embedder, allow_dangerous_deserialization=True)
        retriever = store.as_retriever()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Answer solely based on the following context:\n<Documents>\n{context}\n</Documents>",
                ),
                ("user", "{question}"),
            ]
        )
        self.retriever = retriever
        self.prompt = prompt
