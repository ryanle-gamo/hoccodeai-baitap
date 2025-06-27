import gradio as gr
import base64
import requests

URL = "http://127.0.0.1:7860"

selected_steps=25
selected_guidence_scale=7.5
negative_prompt="worst quality, low quality, watermark, text, error, blurry, jpeg artifacts, cropped, jpeg artifacts, signature, watermark, username, artist name, bad anatomy"

def generate_image(prompt, height, width):
    payload = {
        "prompt": f"{prompt}",
        "negative_prompt":f"{negative_prompt}",
        "steps": selected_steps,
        "cfg_scale": selected_guidence_scale,
        "width": float(width),
        "height": float(height),
    }
    print(payload)
    response = requests.post(f"{URL}/sdapi/v1/txt2img", json=payload)
    resp_json = response.json()
    print("Inference Completed")
    for i, img in enumerate(resp_json['images']):
        print(f"Saving image output_image_{i}.png")
        base64_to_image(img, f"output_image_{i}.png")

def base64_to_image(base64_string, save_path='output_image.png'):
    with open(save_path, 'wb') as f:
        f.write(base64.b64decode(base64_string))

# Định nghĩa giao diện Gradio
with gr.Blocks() as demo:
    # Tạo một hàng chứa tất cả các phần tử
    gr.Markdown("# Create your Image")
    with gr.Row():
        # Tạo một cột bên trái cho các hộp văn bản
        with gr.Column():
            prompt = gr.Textbox(
                label="Prompt", placeholder="Prompt")
            height = gr.Textbox(label="Height", placeholder="Enter image's height")
            width = gr.Textbox(label="Width", placeholder="Enter image's width")

            generate_button = gr.Button("Generate")

    # Kết nối event khi bấm vào nút "Tạo ảnh"
    generate_button.click(fn=generate_image, inputs=[prompt, height, width])

# Chạy giao diện Gradio bằng hàm launch()
demo.launch()