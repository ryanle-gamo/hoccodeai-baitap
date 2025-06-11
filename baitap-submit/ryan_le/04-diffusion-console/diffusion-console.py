import gradio as gr
from diffusers import DiffusionPipeline, EulerDiscreteScheduler, DDIMScheduler, DPMSolverMultistepScheduler
import torch


negative_prompt = ""
guidance_scale = 6.5
num_inference_steps=24

pipeline = DiffusionPipeline.from_pretrained("stablediffusionapi/anything-v5",
                                             use_safetensors=True, safety_checker=None, requires_safety_checker=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline.to(device)
print(pipeline)


def image_generator(prompt, width, height):
    image = pipeline(
    prompt,
    height=float(height),
    width=float(width),
    guidance_scale=guidance_scale,
    num_inference_steps=num_inference_steps,
    negative_prompt=negative_prompt,
    generator=torch.Generator(device=device).manual_seed(6969)).images[0]
    image.save("demo.png")

with gr.Blocks() as demo:
    prompt = gr.Textbox(label="Enter your prompt:")
    width = gr.Textbox(label="Image's width:")
    height = gr.Textbox(label="Image's height:")
    generate_button = gr.Button("Generate")
    generate_button.click(image_generator, [prompt, width, height])

demo.launch(share=True)