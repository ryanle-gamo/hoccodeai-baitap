import weaviate
import kagglehub
from kagglehub import KaggleDatasetAdapter
from weaviate.classes.config import Configure, Property, DataType, Tokenization
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

COMPLETION_MODEL = os.getenv("COMPLETION_MODEL")
COLLECTION_NAME = "BOOKS_DATA1"
KAGGLE_CSV_FILE = "commonlit_texts.csv"

class DataProcessing:

    def __init__(self, db_client: weaviate.WeaviateClient):
        # Public attributes
        self.db_client = db_client

    # Hàm này dùng để xoá tất cả db trong weaviate.
    def delete_all_dbs(self):
        self.db_client.collections.delete_all()
        self.db_client.close()
        print("All dbs have been deleted.")

    def get_kaggle_data(self):
        # Load the latest version
        data = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "kononenko/commonlit-texts",
        KAGGLE_CSV_FILE
        )
        # Validate data trước khí chuyển lên weaviate
        sent_to_vector_db = data.to_dict(orient='records')
        for item in sent_to_vector_db:
            if np.isnan(item["lexile"]):
                item["lexile"] = 0
        return sent_to_vector_db

    def create_collection(self):
        # Tạo schema cho collection
        book_collection = self.db_client.collections.create(
            name=COLLECTION_NAME,
            # Sử dụng model transformers để tạo vector
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(),
            generative_config=Configure.Generative.openai(
                model=COMPLETION_MODEL,
                max_tokens=500,
                presence_penalty=0,
                temperature=0.7,
                top_p=0.7
            ),
            properties=[
                # Tiêu đề sách: text, được vector hóa và chuyển thành chữ thường
                Property(name="title", 
                        data_type=DataType.TEXT,
                        vectorize_property_name=True, 
                        tokenization=Tokenization.LOWERCASE),
                Property(name="author", 
                        data_type=DataType.TEXT, 
                        vectorize_property_name=True, 
                        tokenization=Tokenization.LOWERCASE),
                Property(name="description", 
                        data_type=DataType.TEXT, 
                        vectorize_property_name=True, 
                        tokenization=Tokenization.LOWERCASE),
                Property(name="grade", 
                        data_type=DataType.INT, 
                        vectorize_property_name=True),
                Property(name="genre", 
                        data_type=DataType.TEXT, 
                        tokenization=Tokenization.WORD),
                Property(name="lexile", 
                        data_type=DataType.INT, 
                        skip_vectorization=True),
                Property(name="path", 
                        data_type=DataType.TEXT, 
                        skip_vectorization=True, 
                        tokenization=Tokenization.WHITESPACE),
                Property(name="is_prose", 
                        data_type=DataType.INT, 
                        skip_vectorization=True),
                Property(name="date", 
                        data_type=DataType.TEXT, 
                        vectorize_property_name=True),
                Property(name="intro", 
                        data_type=DataType.TEXT, 
                        vectorize_property_name=True, 
                        tokenization=Tokenization.LOWERCASE),
                Property(name="excerpt", 
                        data_type=DataType.TEXT, 
                        vectorize_property_name=True, 
                        tokenization=Tokenization.LOWERCASE),
                Property(name="license", 
                        data_type=DataType.TEXT, 
                        vectorize_property_name=True, 
                        tokenization=Tokenization.LOWERCASE),
                Property(name="notes", 
                        data_type=DataType.TEXT, 
                        vectorize_property_name=True, 
                        tokenization=Tokenization.LOWERCASE)
            ]
        )
        
        # Import dữ liệu vào DB theo batch
        with book_collection.batch.dynamic() as batch:
            sent_to_vector_db = self.get_kaggle_data()
            for data_row in sent_to_vector_db:
                print(f"Inserting: {data_row['title']}")
                batch.add_object(properties=data_row)

        print("Data saved to Vector DB")

    def search_near_text(self, query: str):
        books_collection = self.db_client.collections.get(COLLECTION_NAME)
        response = books_collection.generate.near_text(
            query= query, 
            single_prompt="""
Create a brief about book: {title}, author: {author}
""",
            limit=5
        )
        return response
    
    def start_db_generation(self):
        if self.db_client.collections.exists(COLLECTION_NAME):
            print("Collection {} already exists".format(COLLECTION_NAME))
            # self.db_client.collections.delete_all()
        else:
            self.create_collection()
            print("DB is ready: {}".format(self.db_client.is_ready()))