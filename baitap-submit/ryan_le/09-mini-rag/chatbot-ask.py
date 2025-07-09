# 2. Thay vì hardcode `doc = wiki.page('Hayao_Miyazaki').text`, sử dụng function calling để:
#   - Lấy thông tin cần tìm từ câu hỏi
#   - Dùng `wiki.page` để lấy thông tin về
#   - Sử dụng RAG để có kết quả trả lời đúng.

import os
from openai import OpenAI
import json
import inspect
import chromadb
from chromadb.utils import embedding_functions
from wikipediaapi import Wikipedia

from pydantic import TypeAdapter
from dotenv import load_dotenv

load_dotenv()
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME") #Enhancement: COLLECTION_NAME should be dynamic to be able to call for different celebrities.
EMBEDDING_FUNCTION = embedding_functions.DefaultEmbeddingFunction()
CLIENT_DATA = chromadb.PersistentClient(path=f"./data")
CLIENT_AI = OpenAI(api_key=OPEN_API_KEY)

SYSTEM_PROMPT = """
You are professional assistant in searching celebrity's information. 
You need to answer with given data, not getting from other side (google, website,...)
With the user's question, you need to find correct name of the celebrity, then use this name to get detail information of selected celebrity.
"""

def is_collection_existed(collection_name: str) -> bool:
    collections = CLIENT_DATA.list_collections()
    for item in collections:
        if item.name.lower() == collection_name.lower():
            return True
    return False

def prepare_data(celebrity_name: str) -> bool:
    """
    Prepare data about selected celebrity, there are two cases:
    1. Existed: retrieve and continue.
    2. Not existed: call Wekipedia and store in db
    """
    if is_collection_existed(COLLECTION_NAME) == False:
        wiki = Wikipedia('HocCodeAI/0.0 (https://hoccodeai.com)', 'en')
        celebrity_info = wiki.page(celebrity_name)
        paragraphs = celebrity_info.text.split('\n\n')
        try:
            collection = CLIENT_DATA.create_collection(
                name=COLLECTION_NAME, 
                embedding_function=EMBEDDING_FUNCTION
            )
            for index, paragraphs in enumerate(paragraphs):
                print(f"{index} |====| {paragraphs}\n")
                collection.add(documents=[paragraphs], ids=[str(index)])
            return True
        except ValueError:
            print(f"ERROR\n")
            return False
    return True

def get_celebrity_detail(celebrity_name: str, question: str) -> str:
    """
    Get detail information of selected celebrity from given data.
    param celebrity_name: The name of the celebrity.
    param question: User's question.
    output: a text to request final answer.
    """
    data_preparation_result = prepare_data(celebrity_name)
    if data_preparation_result == True:
        print("START CALLING FINAL ANSWER\n")
        collection = CLIENT_DATA.get_collection(
            name=COLLECTION_NAME, 
            embedding_function=EMBEDDING_FUNCTION
        )
        q = collection.query(query_texts=[question], n_results=3)
        context = q["documents"][0]
        return f"""
            Use the following CONTEXT to answer the QUESTION at the end.
            If you do not know the answer, just say that you do not know, do not try to make up an answer.
            Use an unbiased and journalist tone.
            Please answer in **Vietnamese**.

            CONTEXT: {context}

            QUESTION: {question}
            """
    print("NO QUESTION\n")
    return f"""Not existed the data that user requested. Ask user to give another question."""

def get_completion(messages):
    return CLIENT_AI.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=messages,
        tools=tools,
        temperature=0
    )

FUNCTION_MAP = {
    "get_celebrity_detail": get_celebrity_detail,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_celebrity_detail",
            "description": inspect.getdoc(get_celebrity_detail),
            "parameters": TypeAdapter(get_celebrity_detail).json_schema(),
        },
    }
]

messages=[
    {
        "role":"system",
        "content":SYSTEM_PROMPT
    }
]

while True:

    #Bắt đầu đặt câu hỏi
    question = input("Bạn muốn hỏi thông tin gì?=> ")

    #Thoát khỏi hội thoại
    if question.lower() in ['no', 'exit','close']:
        break
    
    messages.append(
            {"role": "user", "content": question}
        )

    #Với request này, tên của celebrity sẽ được tìm thấy
    response = get_completion(messages)
    # print(f"BOT celebrity's name:{response}\n")
    first_choice = response.choices[0]
    finish_reason = first_choice.finish_reason

    while finish_reason != "stop":        
        tool_call = first_choice.message.tool_calls[0]
        # print(f"BOT tool_call_name:{tool_call}\n")

        #Chọn hàm để bắt đầu tìm kiếm dữ liệu
        tool_call_function = tool_call.function
        tool_call_arguments = json.loads(tool_call_function.arguments)

        tool_function = FUNCTION_MAP[tool_call_function.name]
        result = tool_function(**tool_call_arguments)

        messages.append(first_choice.message)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call_function.name,
            "content": json.dumps(result)
        })

        # print(f"BOT final message to request answer:{messages}\n")
        # Chờ kết quả từ LLM
        response = get_completion(messages)
        first_choice = response.choices[0]
        finish_reason = first_choice.finish_reason

    # In ra kết quả sau khi đã thoát khỏi vòng lặp
    bot = first_choice.message.content
    print(f"BOT: {bot}\n")
    messages.append(
        {"role": "assistant", "content": bot}
    )

