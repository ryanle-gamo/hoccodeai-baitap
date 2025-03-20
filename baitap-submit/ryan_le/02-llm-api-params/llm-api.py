1. Viết một ứng dụng console đơn giản.
2. Cải tiến ứng dụng chat.
=====>
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key='TOGETHER_AI_KEY',
)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "developer",
            "content": "You are a farmer. You are planting coconut for selling.",
        },
        {
            "role": "user",
            "content": "I just harvest lots of coconuts, I want to sell it to super market. I put price as the following: 1 coconut is 1.2$, a block of coconuts is 10$. Which type do you prefer me to sell to get highest profit?",
        },
        {
            "role": "assistant",
            "content": "A block will be 12 coconuts"
        }
    ],
    model="meta-llama/Llama-3-70b-chat-hf",
    max_tokens=1000,
    temperature=0.6,
    top_p=0.2,
    stream=True,
    store=True
)
for chunk in chat_completion:
    print(chunk.choices[0].delta.content or "", end="")

<=====
------------------------------------------------
3. Tóm tắt website.
=====>
import os
from openai import OpenAI

import requests
from bs4 import BeautifulSoup

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key='TOGETHER_AI_KEY',
)


# URL of the website you want to scrape
url = "https://tuoitre.vn/cac-nha-khoa-hoc-nga-bao-mat-troi-manh-nhat-20-nam-sap-do-bo-trai-dat-2024051020334196.htm?source=0d84f3"

# Send a GET request to fetch the raw HTML content
response = requests.get(url)

if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Find the div with id "main-detail"
    main_detail = soup.find("div", id="main-detail")
    
    if main_detail:
        # Extract text content
        web_content = main_detail.get_text(separator="\n", strip=True)
        chat_completion = client.chat.completions.create(
          messages=[
              {
                  "role": "developer",
                  "content": "Bạn là một bác sĩ, bạn có nhiều bệnh nhân lớn tuổi.",
              },
              {
                  "role": "user",
                  "content": f"Tiến hành tóm tắt nội dung bài báo: {web_content}, sau đó cung cấp lời khuyên đến các bệnh nhân để hạn chế tác hại của cơn bão.",
              }
          ],
          model="meta-llama/Llama-3-70b-chat-hf",
          max_tokens=1024,
          temperature=0.2,
          top_p=0.2,
          stream=True,
      )
        for chunk in chat_completion:
          print(chunk.choices[0].delta.content or "", end="")
    else:
        print("Content not found.")
else:
    print("Failed to fetch the webpage. Status code:", response.status_code)

<=====
------------------------------------------------
4. Dịch nguyên 1 file dài từ ngôn ngữ này sang ngôn ngữ khác.
======>
import os
from openai import OpenAI
import textwrap

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key='TOGETHER_AI_KEY',
)

# Define the model name
model_name = "meta-llama/Llama-3-70b-chat-hf"

# Step 1: Read the novel file
filename = "novel.txt"

with open(filename, "r", encoding="utf-8") as file:
    novel_content = file.read()

# Step 2: Split the novel into chunks based on model's token limit
MAX_TOKENS = 4096  # Adjust based on TogetherAI's model context limit
chunks = textwrap.wrap(novel_content, width=MAX_TOKENS, break_long_words=True)

# Step 3: Translate each chunk using TogetherAI
translated_parts = []

for i, chunk in enumerate(chunks):
    response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "developer",
            "content": """
            You are a professional translator.
            You need to translate a long text from a given language to others based on user's request.
            Translated text must be accuracy and natural.
            Maintain the original meaning, style, and cultural nuances.
            """,
            "role": "user", "content": f"Translate the following text into English. Output only the translated text—do not include any introductions, explanations, or extra formatting:\n\n{chunk}"}]
    )
    
    translated_text = response.choices[0].message.content
    translated_parts.append(translated_text)
    print(translated_text)
    print(f"Translated part {i+1}/{len(chunks)} completed.")

# Step 4: Save the translated content into a new file
output_filename = "translated_novel.txt"

with open(output_filename, "w", encoding="utf-8") as file:
    file.write("\n\n".join(translated_parts))

print(f"Translation completed! The translated novel is saved as {output_filename}.")
<=====
------------------------------------------------
5. Dùng bot để giải bài tập lập trình.
Bài tập này mình có tham khảo bài của một số bạn khác, sau đó hiệu chỉnh theo ý của mình.
======>
import datetime
import os
from openai import OpenAI

client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key='TOGETHER_AI_KEY',
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "developer",
            "content": """
You are a professional Python programmer.
Your task is to write Python code that solves the problems are presented from user.
Do not explain the code, just generate the code itself.
Do not attach quotes.

Ensure your code is efficient, well-commented, and handles edge cases appropriately.  Prioritize clarity and readability in your solutions.
The input will consist of a problem description.
Carefully analyze the input to understand the problem thoroughly before writing any code.
Your code should be robust enough to handle various inputs related to the described problem, not just the specific examples provided.
Consider edge cases and potential errors, and implement appropriate error handling mechanisms.

Your Python code should be written within the main() function.
If the problem requires external libraries, import them at the beginning.

Here are some additional points to consider when writing your code:
- Clarity: Use meaningful variable names and follow Python PEP 8 – Style Guide for readability.
- Efficiency: Optimize your code for performance, especially if the problem involves large datasets or complex computations.
- Error Handling: Implement appropriate error handling to gracefully manage unexpected inputs or situations. Use try-except blocks where necessary.
- Testability: Write code that is easy to test and debug. Consider including simple test cases within comments to demonstrate the functionality of your code.
- Modularity: If the problem can be broken down into smaller sub-problems, consider creating separate functions for each sub-problem to improve code organization and reusability.
- Comments: Explain the purpose of your code, the logic behind your approach, and any assumptions you've made. Clear comments are crucial for understanding and maintaining your code.
- Input Validation: Validate the input to ensure it meets the specified requirements and constraints. Handle invalid input gracefully.
            """,
        },
        {
            "role": "user",
            "content": "I would like to create a calculation program with 4 operations: addition, subtraction, multiplication, division.",
        }
    ],
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0.2,
    max_tokens=1000,
)
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"./final_{current_time}.py"
with open(filename, "w", encoding="utf-8") as f:
  content = response.choices[0].message.content
  print(content)
  f.write(content)
<=======
------------------------------------------------
