import asyncio
import os
import faiss
import shutil
import numpy as np
from elasticsearch import AsyncElasticsearch
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from sentence_transformers import SentenceTransformer
import gradio as gr
from data_processing1 import get_chunk_data
import numpy as np
from sentence_transformers import CrossEncoder
from langchain_openai import ChatOpenAI
import glob
import tracemalloc
tracemalloc.start()
# Initialize API Keys securely
openai_Key = "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"
os.environ["OPENAI_API_KEY"] = openai_Key
# Set the Hugging Face API key in the environment (if needed)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


#cross encoder reranker
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Initialize the model for creating embeddings (Sentence-Transformer)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Elasticsearch connection
es = AsyncElasticsearch(
    hosts=["http://localhost:9200"]
)

# ElasticsearchStore
es_store = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="pdf_documents",
    client=es
)

uploaded_files = []
UPLOAD_FOLDER = "./data1"

async def ensure_upload_folder(upload_folder_path):
    # Check if the upload folder exists
    if os.path.exists(upload_folder_path):
        # If it exists, delete all files inside it
        for filename in os.listdir(upload_folder_path):
            file_path = os.path.join(upload_folder_path, filename)
            try:
                # Delete the file or folder
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
    else:
        # If it doesn't exist, create the folder
        os.makedirs(upload_folder_path)
        print(f"Created folder: {upload_folder_path}")

# Example usage

# Step 1.1: Delete the existing Elasticsearch index

# Step 1.2: Delete all Elasticsearch indices
async def delete_all_indices(es_store):
    # Check if the specific index exists
    if await es_store.client.indices.exists(index="pdf_documents"):
        # Delete the specified index
        await es_store.client.indices.delete(index="pdf_documents")
        print("Index 'pdf_documents' deleted.")
    else:
        print("Index 'pdf_documents' does not exist.")
    close_client()
# Step 9: Close Elasticsearch client session
async def close_client():
    await es_store.client.close()
    print("Elasticsearch client closed.")
# FAISS index for vector search 
faiss_index = None

# Step 1: Create an index for PDF documents if it doesn't exist in Elasticsearch
async def create_index():
    es_store = ElasticsearchStore(
    es_url="http://localhost:9200",
    index_name="pdf_documents",
    client=es
)

    if not await es_store.client.indices.exists(index=es_store.index_name):
        await es_store.client.indices.create(index=es_store.index_name, body={
            'mappings': {
                'properties': {
                    'text_field': {
                        'type': 'text'
                    }
                }
            }
        })
        print(f"Index '{es_store.index_name}' created.")

def read_and_embed_pdf_in_chunks(file_path):
    text_chunks = get_chunk_data(file_path)
    print("text_chunks",text_chunks)
    embeddings = []

    # Generate embeddings in small batches if the data is large
    for chunk in text_chunks:
        embedding = model.encode(chunk, convert_to_tensor=False)
        embeddings.append(embedding)

    return text_chunks, np.array(embeddings)

# Step 3: Index the PDF chunks into Elasticsearch and FAISS
async def index_pdf_chunks_and_embeddings(pdf_file_paths):
    """Reads multiple PDFs, indexes them in Elasticsearch, and adds embeddings to FAISS."""
    global faiss_index
    global all_text_chunks
    doc_id = 1  # Unique ID for each document chunk

    all_embeddings = []
    all_text_chunks = []

    for pdf_file_path in pdf_file_paths:
        print(f"Processing '{pdf_file_path}'...")
        text_chunks, embeddings = read_and_embed_pdf_in_chunks(pdf_file_path)
        print("text_chunks",text_chunks)
        
        # Index each chunk into Elasticsearch and collect embeddings
        for idx, chunk in enumerate(text_chunks):
            await es_store.client.index(index=es_store.index_name, id=f"{doc_id}-{idx}", body={"text_field": chunk})
            all_text_chunks.append(chunk)
            all_embeddings.append(embeddings[idx])

        doc_id += 1

    # Convert all embeddings to a numpy array
    all_embeddings = np.array(all_embeddings)

    # Create a FAISS index
    embedding_dim = all_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(embedding_dim)  # Using L2 distance (Euclidean distance)
    faiss_index.add(all_embeddings)  # Add embeddings to the FAISS index

# Step 4: Fuzzy search function in Elasticsearch
async def fuzzy_search(query, fuzziness='AUTO', size=3):
    search_body = {
        'query': {
            'match': {
                'text_field': {
                    'query': query,
                    'fuzziness': fuzziness  # Enables partial matching (fuzzy search)
                }
            }
        },
        'size': size  # Return the top 'size' results
    }

    # Perform the search asynchronously
    # response = await es_store.client.search(index=es_store.index_name, body=search_body)
    # return response['hits']['hits'][:size]  # Return only the top 'size' results

    response = await es_store.client.search(index=es_store.index_name, body=search_body)
    return response["hits"]["hits"]

# Step 5: FAISS vector search function
def faiss_search(query, top_k=3):
    """Perform vector-based similarity search using FAISS."""
    query_embedding = model.encode(query, convert_to_tensor=False)
    query_embedding = np.array([query_embedding])  # Make it 2D

    # Perform FAISS search
    # distances, indices = faiss_index.search(query_embedding, top_k)
    
    # Return indices and distances of the closest chunks
    # return indices[0], distances[0]
    # Check if the faiss_index exists before performing the search
    if faiss_index is not None:
        try:
            distances, indices = faiss_index.search(query_embedding, top_k)
            if indices[0][0] == -1:
                return "No data retrieved"
            else:
                print("Data retrieved:", indices, distances)
                return indices[0], distances[0]

        except Exception as e:
            print(f"Error during search: {e}")
            return f"Error during search: {e}",""

    else:
        print("FAISS index is not created. No search can be performed. Ask user to upload the document and search")
        return "FAISS index is not created. No search can be performed. Ask user to upload the document and search", ""

# ranking
def rerank_results(query, resp):
    pairs = [[query, doc_text] for doc_text in resp]
    scores = cross_encoder.predict(pairs)  # Ensure batching for efficiency

    ranked_indices = np.argsort(scores)[::-1]
    ranked_results = [resp[i] for i in ranked_indices]

    return ranked_results

# Step 6: Combine FAISS and Elasticsearch results
async def hybrid_search(query): 
    # Perform fuzzy search
    es_results = await fuzzy_search(query)
    # print("es_results")
    # print(es_results)

    # Perform vector search
    faiss_indices, faiss_distances = faiss_search(query)
    # print("faiss_indices")
    # print(faiss_indices)

    # Combine results
    response = "Hybrid Search Results:\n\n"
    resp = []

    # Add Elasticsearch results
    response += "Elasticsearch Fuzzy Search Results:\n"
    for idx, result in enumerate(es_results, start=1):
        response += f"{idx}. {result['_source']['text_field']}\n"
        resp.append(result['_source']['text_field'])  # Append the actual text instead of formatted string
    
    # Add FAISS results
    response += "\nFAISS Vector Search Results:\n"
    for idx, faiss_index in enumerate(faiss_indices, start=1):
        # Retrieve the actual text chunk using the faiss_index
        faiss_text_chunk = all_text_chunks[faiss_index]  # Ensure all_text_chunks is defined and accessible
        response += f"{idx}. {faiss_text_chunk} (FAISS Index: {faiss_index})\n"
        resp.append(faiss_text_chunk)  # Append the actual text instead of using result from ES

    # Optional: If you want to rerank results based on some criteria
    ranked_results = rerank_results(query, resp)
    
    # Print the combined results
    print("Hybrid Search Results:\n", response)
    print("Reranked Results:\n", ranked_results)

    return response  # You might want to return the response if needed elsewhere


    return ranked_results

# Step 7: Gradio chatbot interface
async def chatbot_llm(query):
    option_1 = os.listdir(UPLOAD_FOLDER)
    if (len(option_1) ==0) | (query==""):
        prompt = f" You are a AI assistand Name called Swarmx Docubot Foe Document RAG. U can read pdf image, docx, txt, csv, excel files. Ask them to upload the Data to retrieve the meaningfull information. if the the query is based on you answer WHAT THEY ARE ASKING. othern then this ask to uplod the data.  Answer the Given the following \n\nQuery: {query}"
    else:
        hybrid_results = await hybrid_search(query)
        context =  hybrid_results
        prompt = f" You are a AI assistand Foe Document RAG. Answer the Given the following context.  given context:\n\n{context}\n\nQuery: {query}"

        print('query')
        print(query)
        print("context")
        # print(context)
    llm = ChatOpenAI(model="gpt-4o")  # Ensure the model name is correct

    response = llm(prompt)
    # print(response)
    return response.content


# Step 8: Initialize indexing and start the chatbot
async def initialize():
    path = "data1"
 # Delete the Elasticsearch index (if it exists)
    # await delete_document_reader_index()
    # Create the Elasticsearch index
    await create_index()

    # List all PDF file paths to process (assuming the data is in text format)
    pdf_file_paths = []
    for x in os.listdir(path):
        # if x.endswith(".pdf"):  # Use actual PDF reading if applicable
        pdf_file_paths.append(os.path.join(path, x))

    # Index the PDF files in Elasticsearch and FAISS
    await index_pdf_chunks_and_embeddings(pdf_file_paths)

    print("PDFs indexed successfully in Elasticsearch and FAISS.")


def delete_files_in_directory(directory_path):
    try:
        files = glob.glob(os.path.join(directory_path, '*'))
        for file in files:
            if os.path.isfile(file):
                os.remove(file)
        print("All files deleted successfully.")
    except OSError:
        print("Error occurred while deleting files.")

if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)

async def upload_file(files):
    global uploaded_files
    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    # Clear old files before uploading new ones
    delete_files_in_directory(UPLOAD_FOLDER)

    uploaded_files = []
    for file in files:
        shutil.copy(file.name, UPLOAD_FOLDER)  # Copy the file
        uploaded_files.append(os.path.basename(file.name))  # Store only file names

    if len(uploaded_files) == 0:
        return "It's empty", []

    print("Files uploaded...")
    await delete_all_indices(es_store)
    close_client
    await initialize()  # Correctly calling the async function

    # return f"Uploaded {len(uploaded_files)} file(s) successfully!"
    return f"{len(files)} file(s) uploaded successfully.", []

def delete_files():
    """Deletes all files in the upload folder."""
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)  # Remove the entire directory and its contents
        os.mkdir(UPLOAD_FOLDER)  # Recreate the empty directory
        return "All files have been deleted."
    return "No files to delete."

def llm_chatbot(message, history):
    
    """Simulates a simple chatbot response."""
    if message:
        print("message-----", message)
        response = "hai"
        return response
    return "Please ask a question."

def show_uploaded_files():
    """Returns the names of the uploaded files."""
    global uploaded_files

    # uploaded_files = os.listdir(UPLOAD_FOLDER)
    # if not uploaded_files:
    #     return "No files have been uploaded."
    # # return "\n".join(uploaded_files)
    # return uploaded_files

    if  uploaded_files:
        return "\n".join(uploaded_files)
    else:
        return "No files have been uploaded."

# Create the Gradio interface with both upload and chat features
# Gradio interface definition
async def main():
    # await initialize()  # Initialize the system
    await ensure_upload_folder(UPLOAD_FOLDER)

    # Define the Gradio chatbot interface
    with gr.Blocks() as demo:  # Using gr.Blocks for flexible layout with chatbot
        gr.Markdown(
            """# Swarmx DocuChat
            Talk with your Document with Swarmx DocuBot!
            """
        )
        chatbot = gr.Chatbot(height=400, placeholder="<strong>Communicate with Swarmaxbot</strong><br>Upload Your files")
        textbox = gr.Textbox(placeholder="Ask me anything", show_label=False,container=False)

        with gr.Row():
            # File Upload Section
            upload_button = gr.UploadButton("Upload Files", file_count="multiple", label="Upload Files")
            # Output display for uploaded file names
            upload_status = gr.Textbox(label="Upload Status", interactive=False)

            # Button to show uploaded files
            show_status_button = gr.Button("Show Uploaded Files")
            show_status_button.click(show_uploaded_files, None, upload_status)

            # Handle file upload and update outputs
            # upload_button.upload(upload_file, upload_button, upload_status)
            upload_button.upload(upload_file, upload_button, [upload_status, chatbot])

        # Chain inference logic: when user submits, call chatbot_llm asynchronously
            async def user_interaction(user_input, history):
                bot_response = await chatbot_llm(user_input)
                history.append((user_input, bot_response))
                return history, ""

        # textbox.submit(user_interaction, inputs=[textbox, chatbot], outputs=[chatbot])
        textbox.submit(user_interaction, inputs=[textbox, chatbot], outputs=[chatbot, textbox])

    # Launch Gradio app
    
    demo.launch(share=True)

# Step 9: Close Elasticsearch client session
async def close_client():
    await es_store.client.close()
    print("Elasticsearch client closed.")

# Run the main function
if __name__ == '__main__':
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(main())    
    finally:
        asyncio.run(close_client())
      