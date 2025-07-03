import os
from openai import OpenAI
import requests
import json
import inspect
from pydantic import TypeAdapter
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
COMPLETION_MODEL = os.getenv("COMPLETION_MODEL")
client = OpenAI(api_key=OPEN_API_KEY)

def get_symbol(company: str) -> str:
    """
    Retrieve the stock symbol for a specified company using the Yahoo API.
    :param company: The name if the company for which to retrieve the stock symbol, e.g, 'Nvidia'.
    :output: The stock symbol for the specified company.
    """
    url = "https://query2.finance.yahoo.com/v1/finance/search"
    params = {"q": company, "country": "United States"}
    user_agents = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/42.0.2311.135 Safari/537.36 Edge/12.246"}
    res = requests.get(
        url=url,
        params=params,
        headers=user_agents
    )
    data = res.json()
    symbol = data['quotes'][0]['symbol']
    return symbol

def get_stock_price(symbol:str):
    """
    Retrieve the most recent stock price data for a specified company using the Yahoo Finance API via yfinance Python library.
    :param symbol: The stock symbol for which to retrieve data, e.g., 'NVDA' for Nvidia.
    :output: A dictionary containing the most recent stock price data.
    """
    stock = yf.Ticker(symbol)
    hist = stock.history(period="1d", interval='1m')
    latest = hist.iloc[-1]
    return {
        "timestamp": str(latest.name),
        "open": latest["Open"],
        "high": latest["High"],
        "low": latest["Low"],
        "close": latest["Close"],
        "volume": latest["Volume"],
    }

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_symbol",
            "description": inspect.getdoc(get_symbol),
            "parameters": TypeAdapter(get_symbol).json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": inspect.getdoc(get_stock_price),
            "parameters": TypeAdapter(get_stock_price).json_schema(),
        },
    }
]

FUNCTION_MAP = {
        "get_symbol": get_symbol,
        "get_stock_price": get_stock_price
        }

def get_completion(messages):
    response = client.chat.completions.create(
        model=COMPLETION_MODEL,
        messages=messages,
        tools=tools,
        temperature=0 #most stable result
    )
    return response

# Khởi tạo LLM với các định nghĩa ban đầu.
messages = [
    {
        "role":"system",
        "content":"You are helpful customer support assistant. Use the supplied tools to assist the user. You know Trump president is quite not abnormal, you will provide more notice for user if they want to invest the stock"
    }
]

while True:
    question = input("Có câu hỏi gì không bạn ơi? => ")

    #Thoát khỏi hội thoại
    if question.lower() in ['no', 'exit','close']:
        break

    messages.append(
        {
            "role":"user",
            "content":question
        }
    )

    #Gửi câu hỏi lên LLM
    response = get_completion(messages)
    print(f"1: {response}")

    # LLM trả kết quả về, nếu finish_reason là STOP => vào kiểm tra xem function nào sẽ được gọi
    first_choice = response.choices[0]
    finish_reason = first_choice.finish_reason

    # Vòng lặp chạy đến khi finish_reason = STOP
    while finish_reason != "stop":

        tool_call = first_choice.message.tool_calls[0]

        # Lấy tên hàm và các biến
        tool_call_function = tool_call.function
        tool_call_arguments = json.loads(tool_call_function.arguments)

        # Check name để chọn hàm
        tool_function = FUNCTION_MAP[tool_call_function.name]
        result = tool_function(**tool_call_arguments)
        print(f"2: {result}")
        
        # Gửi hàm cùng tham số lên LLM
        messages.append(first_choice.message)
        messages.append(
            {
                "role":"tool",
                "tool_call_id":tool_call.id,
                "name":tool_call_function.name,
                "content":json.dumps(result)
            }
        )

        #Sau khi phân tích tham số, đưa messages cho LLM chạy tiếp
        response = get_completion(messages)
        print(f"3: {messages}")
        first_choice = response.choices[0]
        finish_reason = first_choice.finish_reason

    # In ra kết quả cho người dùng
    bot_message = first_choice.message.content
    print(f"Advisor: {bot_message}")

    # Thêm kết quả vào messages để LLM nhớ cho lần hỏi sau
    messages = [
        {
            "role":"assistant",
            "content":bot_message
        }
    ]