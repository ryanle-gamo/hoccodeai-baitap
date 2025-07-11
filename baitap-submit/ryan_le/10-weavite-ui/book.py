import gradio as gr
import weaviate
from weaviate.embedded import EmbeddedOptions

from insert_data import DataProcessing
from insert_data import COLLECTION_NAME

HEADERS = [
    "title", "author", "description", "grade", "genre", "lexile",
    "path", "is_prose", "date", "intro", "excerpt", "license", "notes"
]

# Cần chạy docker container cho model embedding:
# docker run -itp "8000:8080" semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
embedded_options = EmbeddedOptions(
    additional_env_vars={
        # Kích hoạt các module cần thiết: text2vec-transformers
        "ENABLE_MODULES": "backup-filesystem,text2vec-transformers",
        "BACKUP_FILESYSTEM_PATH": "/tmp/backups",  # Chỉ định thư mục backup
        "LOG_LEVEL": "panic",  # Chỉ định level log, chỉ log khi có lỗi
        "TRANSFORMERS_INFERENCE_API": "http://localhost:8000"  # API của model embedding
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
    response = data_processing.search_near_text(query, 10)
    # Trả về author và title của các sách liên quan
    results = []
    for book in response.objects:
        book_tuple = (book.properties['title'], book.properties['author'], book.properties['description'], 
                    book.properties['grade'], book.properties['genre'], book.properties['lexile'], 
                    book.properties['path'], book.properties['is_prose'], book.properties['date'], 
                    book.properties['intro'], book.properties['excerpt'], book.properties['license'], book.properties['notes'])
        results.append(book_tuple)
    print(results)
    return results

def show_result_with_key(search_key):
    data = search_book(search_key)
    return gr.Dataframe(value=data, headers=HEADERS, datatype="str", label="Tìm kiếm sách")

with gr.Blocks(title="Tìm kiếm sách với Vector Database") as interface:
    query = gr.Textbox(label="Tìm kiếm sách", placeholder="Tên, tác giả, thể loại, năm xuất bản,...")
    search = gr.Button(value="Search")
    results = gr.Dataframe(headers=HEADERS, label="DANH SÁCH KẾT QUẢ"),
    # Khi người dùng bấm search, ta gọi hàm search_book với đầu vào là query và truyền kết quả vào results
    search.click(fn=show_result_with_key, inputs=query, outputs=results)

interface.queue().launch()
