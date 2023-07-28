from potassium import Potassium, Request, Response
import torch
import base64
from io import BytesIO
from diffusers import DiffusionPipeline

app = Potassium("my_app")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
    )
    base.to("cuda")
    base.enable_xformers_memory_efficient_attention()

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.to("cuda")

    context = {
        "model": base,
        "refiner": refiner
    }

    return context

# @app.handler runs for every call
@app.handler("/")
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    num_steps = request.json.get("num_steps", 50)
    high_noise_frac = request.json.get("high_noise_frac", 0.8)

    model = context.get("model")
    refiner = context.get("refiner")
    image = model(
        prompt=prompt,
        num_inference_steps=num_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
    ).images

    image = refiner(
        prompt=prompt,
        num_inference_steps=num_steps,
        denoising_start=high_noise_frac,
        image=image,
    ).images[0]

    buffered  = BytesIO()
    image.save(buffered ,format="JPEG")
    output = base64.b64encode(buffered .getvalue()).decode('utf-8')

    return Response(
        json = {"output": output}, 
        status=200
    )

if __name__ == "__main__":
    app.serve()