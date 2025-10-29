from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
import glob
from langchain_ollama import ChatOllama
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import uuid
from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
import base64
from IPython.display import Image, display
import pickle

load_dotenv()


def read_pdf_files_from_directory(directory_path:str) -> list:
    
    list_of_elements = []

    files = glob.glob(directory_path + "/*.pdf")

    for file_path in files:
        chunks = partition_pdf(
            filename=file_path,
            languages=["ara", "eng"],
            infer_table_structure=True,
            strategy="hi_res",

            extract_image_block_types=["Image"],

            extract_image_block_to_payload=True,

            chunking_strategy="basic",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )
        print(len(chunks))
        list_of_elements.extend(chunks)
    print('Finished reading all PDF files.')
    return list_of_elements


# separate tables from texts
def separate_tables_and_texts(list_of_elements:list):
    tables = []
    texts = []
    
    for chunk in list_of_elements:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk)
    print('Finished separating tables and texts.')
    return tables, texts


# Get the images from the CompositeElement objects
def get_images_base64(chunks):
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    print(f'Extracted {len(images_b64)} images from the document chunks.')
    return images_b64


def display_base64_image(base64_code):
    # Decode the base64 string to binary
    image_data = base64.b64decode(base64_code)
    # Display the image
    display(Image(data=image_data))


def summarize_chain(model:str="llama-3.1-8b-instant"):
    # Prompt
    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    # Summary chain
    model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()
    return summarize_chain


def describe_images(images:list):
    prompt_template = """Describe the image in detail. For context, the image is part of a power point presentation to the insurance service, GIG. Be specific about images and other details."""
    messages = [
        (
            "user",
            [
                {"type": "text", "text": prompt_template},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,{image}"},
                },
            ],
        )
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    chain = prompt | ChatOllama(
        model="llava:latest", 
        base_url=os.getenv("OLLAMA_BASE_URL")
        ) | StrOutputParser()

    if os.path.exists('image_summaries.pkl'):
        pickle_file = 'image_summaries.pkl'
        with open(pickle_file, 'rb') as f:
            image_summaries = pickle.load(f)
        return image_summaries
    else:
        image_summaries = chain.batch(images)
        with open('image_summaries.pkl', 'wb') as f:
            pickle.dump(image_summaries, f)
        return image_summaries

if __name__ == "__main__":

    embeddings = OllamaEmbeddings(
        model="embeddinggemma:latest", 
        base_url=os.getenv("OLLAMA_BASE_URL")
        )

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        collection_name="multi_modal_rag", 
        embedding_function=embeddings, 
        persist_directory='./data/chroma_db'
        )
                        
    store = InMemoryStore()
    id_key = "doc_id"

    retriever = vectorstore.as_retriever()

    chunks = read_pdf_files_from_directory('../data/pdfs')
    tables, texts = separate_tables_and_texts(chunks)

    summarize_chain = summarize_chain()
    
    # Summarize text
    print('Summarizing texts...')
    text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

    # Summarize tables
    print('Summarizing tables...')
    tables_html = [table.metadata.text_as_html for table in tables]
    table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

    images = get_images_base64(chunks)
    image_summaries = describe_images(images)


    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    if summary_texts != []:
        retriever.vectorstore.add_documents(summary_texts)

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    if summary_tables != []:
        retriever.vectorstore.add_documents(summary_tables)

    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in images]
    summary_img = [
        Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
    ]
    if summary_img != []:
        retriever.vectorstore.add_documents(summary_img)
