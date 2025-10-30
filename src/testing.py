import chromadb

client = chromadb.PersistentClient(path="./data/chroma_db")

collection = client.get_collection('multi_modal_rag')

print(collection.get())

with open('chromadb data.txt', 'w', encoding='UTF-8') as f:
    data = f.write(str(collection.get()))