import torch
from diffusers import StableDiffusionPipeline

# -----------------------------------------
# 1. Load Pre-trained Text-to-Image Model
# -----------------------------------------
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# -----------------------------------------
# 2. Engineering Text Prompt
# -----------------------------------------
prompt = """
A futuristic engineering concept of a modern suspension bridge,
large steel cables, strong concrete towers, vehicles crossing the bridge,
advanced structural design, realistic engineering visualization,
daylight, high detail, professional architectural rendering
"""

# -----------------------------------------
# 3. Generate Image
# -----------------------------------------
print("Generating engineering concept image...")

image = pipe(
    prompt,
    num_inference_steps=30
).images[0]

# -----------------------------------------
# 4. Save Image
# -----------------------------------------
image.save("engineering_bridge_concept.png")

print("Image generated successfully!")
print("Saved as: engineering_bridge_concept.png")

# Display image
image.show()