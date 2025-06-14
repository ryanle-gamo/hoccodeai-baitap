import gradio as gr
from diffusers import DiffusionPipeline
import torch

# Kiểm tra thiết bị có GPU hay không
if torch.cuda.is_available():
    device_name = torch.device("cuda")
    torch_dtype = torch.float16
else:
    device_name = torch.device("cpu")
    torch_dtype = torch.float32

# Sử dụng model Dreamshaper từ civit.ai.
# Cần convert file safetensor sang package có thể chạy với thư viện diffusers
# Tutorial ở đây: https://medium.com/@natsunoyuki/using-civitai-models-with-diffusers-package-45e0c475a67e
# Sử dụng thêm 2 models ở Huggingface vào trong Dropdown, người dùng có nhiều lựa chọn.

civit_model="dreamshaper_8/"
models=[civit_model,
        "sd-legacy/stable-diffusion-inpainting",
        "stabilityai/stable-diffusion-xl-base-1.0"]

#Đặt giá trị mặc định cho các thông số
selected_model=models[0]
selected_seed=-1
selected_guidence_scale=7.5
selected_inference_steps=20

def model_selection(selected_value):
    selected_model = selected_value
    print(f"Selected model: {selected_model}")

def seed_selection(selected_value):
    selected_seed = selected_value
    print(f"Selected seed: {selected_seed}")

def guidence_selection(selected_value):
    selected_guidence_scale = selected_value
    print(f"Selected guidence: {selected_guidence_scale}")

def inference_selection(selected_value):
    selected_inference_steps = selected_value
    print(f"Selected inference step: {selected_inference_steps}")

def generate_image(prompt, negative_prompt):
    # Khởi tạo pipeline cho mô hình Stable Diffusion đã được huấn luyện trước.
    pipeline = DiffusionPipeline.from_pretrained(selected_model, 
                                             torch_dtype=torch_dtype,
                                             use_safetensors=False, safety_checker = None)
    pipeline.to(device_name)  
    
    generated_images = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=512,
        seed=selected_seed,
        guidance_scale=selected_guidence_scale,
        num_inference_steps=selected_inference_steps
    ).images
    return generated_images[0]

# Định nghĩa giao diện Gradio
with gr.Blocks() as demo:
    # Tạo một hàng chứa tất cả các phần tử
    gr.Markdown("# Create your Image")
    with gr.Row():
        # Tạo một cột bên trái cho các hộp văn bản
        with gr.Column():
            model_dropbox = gr.Dropdown(choices=models, label="Choose model", interactive=True)
            model_dropbox.change(fn=model_selection, inputs=model_dropbox)

            prompt = gr.Textbox(
                label="Prompt", placeholder="Prompt")
            negative_prompt = gr.Textbox(
                label="Negative Prompt", value="ugly, deformed, low quality", placeholder="Negative prompt")
            
            seed = gr.Number(label="Seed", value=-1)
            seed.change(fn=seed_selection, inputs=seed)

            inference_step = gr.Slider(minimum=1, maximum=50, step=1, value=20, label="Inference Steps", info="Choose between 1 and 50", visible=True)
            inference_step.change(fn=inference_selection, inputs=inference_step)

            guidance_scale = gr.Number(label="Guidence Scale", value=7.5, step=0.5)
            guidance_scale.change(fn=guidence_selection, inputs=guidance_scale)

            generate_button = gr.Button("Generate")

        # Tạo một cột bên phải để hiển thị hình ảnh đầu ra
        with gr.Column():
            image_output = gr.Image(label="Output Image", height=512, width=512)

    # Kết nối event khi bấm vào nút "Tạo ảnh"
    generate_button.click(fn=generate_image, inputs=[prompt, negative_prompt], outputs=image_output)

# Chạy giao diện Gradio bằng hàm launch()
demo.launch()