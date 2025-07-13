import os
import gradio as gr
import weaviate
from weaviate.embedded import EmbeddedOptions
from dotenv import load_dotenv

from data_processing import DataProcessing

load_dotenv()

HEADERS = [
    "title", "author", "description"
]

OPEN_API_KEY = os.getenv("OPEN_API_KEY")
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL")

# Cần chạy docker container cho model embedding:
# docker run -itp "8000:8080" semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
embedded_options = EmbeddedOptions(
    additional_env_vars={
        # Kích hoạt các module cần thiết: text2vec-transformers, generative-openai
        "ENABLE_MODULES": "backup-filesystem,text2vec-transformers, generative-openai",
        "BACKUP_FILESYSTEM_PATH": "/tmp/backups",  # Chỉ định thư mục backup
        "LOG_LEVEL": "panic",  # Chỉ định level log, chỉ log khi có lỗi
        "TRANSFORMERS_INFERENCE_API": "http://localhost:8000",  # API của model embedding,
        "OPENAI_APIKEY": OPEN_API_KEY
    },
    persistence_data_path="data",  # Thư mục lưu dữ liệu
)
 
vector_db_client = weaviate.WeaviateClient(
        embedded_options=embedded_options
    )
    
# Bắt đầu kết nối
vector_db_client.connect()

# Init class insert dữ liệu vào db.
data_processing = DataProcessing(vector_db_client)
data_processing.start_db_generation()

# Tìm kiếm sách trong database, tìm kiểu near_text
def search_book(query):
    # Tìm kiếm theo ngữ nghĩa - NEAR_TEXT
    response = data_processing.search_near_text(query)
    # Trả về author và title của các sách liên quan
    results = []
    for book in response.objects:
        book_tuple = (book.properties['title'], book.properties['author'], book.generative.text)
        print(f"Title: {book.properties['title']}")
        results.append(book_tuple)
    return results

def show_result_with_key(search_key):
    data = search_book(search_key)
    return gr.Dataframe(value=data, headers=HEADERS, datatype="str", label="Tìm kiếm sách")

with gr.Blocks(title="Find a book with your favourite") as interface:
    query = gr.Textbox(label="Input", placeholder="Type of book that you wish, e.g. funny, adventure, children...")
    search = gr.Button(value="Search")
    results = gr.Dataframe(headers=HEADERS, label="RESULTS", wrap=True),
    # Khi người dùng bấm search, ta gọi hàm search_book với đầu vào là query và truyền kết quả vào results
    search.click(fn=show_result_with_key, inputs=query, outputs=results)

interface.queue().launch()