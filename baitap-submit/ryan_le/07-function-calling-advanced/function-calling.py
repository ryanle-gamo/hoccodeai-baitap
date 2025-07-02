import gradio as gr
from openai import OpenAI
import requests
import json
import inspect
from pydantic import TypeAdapter


JINAAI_BEARER_TOKEN = "Bearer xxxx"
COMPLETION_MODEL = "gpt-4o-mini"


def view_website(url: str):
    """
    View a website from url using JinaAI
    :param url: URL of website.
    :output: summarization of website
    """
    the_url = f'https://r.jina.ai/{url}'
    headers = {
        'Authorization': JINAAI_BEARER_TOKEN
    }
    response = requests.get(the_url, headers=headers)
    print(response.text)
    return response.text

tools = [
    {
        "type": "function",
        "function": {
            "name": "view_website",
            "description": "View a website",
            "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
        }
    }
]

client = OpenAI(
    api_key='xxx',
)

view_website_function = {
    "name": "view_website",
    "description": inspect.getdoc(view_website),
    "parameters": TypeAdapter(view_website).json_schema(),
}

def parsing_logic(message):

    messages = [
        {"role": "user", "content": message}
    ]

    print("Bước 1: Gửi message lên cho LLM")
    print(messages)

    response = client.chat.completions.create(
        messages=messages,
        model=COMPLETION_MODEL,
        tools=tools
    )
    print("Bước 2: LLM đọc và phân tích ngữ cảnh LLM")
    print(response)

    print("Bước 3: Lấy kết quả từ LLM")
    tool_call = response.choices[0].message.tool_calls[0]
    print(tool_call)

    print("Bước 4: Chạy function ở máy mình")
    if tool_call.function.name == 'view_website':
        arguments = json.loads(tool_call.function.arguments)
        website_content = view_website(arguments.get('url'))
        print(f"Kết quả bước 4: {website_content}")     

    print("Bước 5: Gửi kết quả lên cho LLM")  
    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "content": website_content,
        "tool_call_id": tool_call.id
    })
    print(messages)

    final_response = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=messages
        # Ở đây không có tools cũng không sao, vì ta không cần gọi nữa
    )
    summarization = final_response.choices[0].message.content
    print(
        f"Kết quả cuối cùng từ LLM: {summarization}.")

    yield "", summarization

with gr.Blocks() as demo:
    gr.Markdown("# Summarize a website")
    message = gr.Textbox(label="Enter website's url:")
    summarization = gr.Textbox(label="Summarization", interactive=False)
    message.submit(parsing_logic, message, summarization)

demo.launch()