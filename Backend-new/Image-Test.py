# Load model directly
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# --- Configuration ---
LOCAL_IMAGE_PATH = "2501.17887v1\_page_4_Figure_2.jpeg" 
#ModelName = "Qwen/Qwen2-VL-7B-Instruct"
ModelName = "HuggingFaceTB/SmolVLM-Instruct"
# ---------------------
# 1. Load model with GPU/CPU Offload and Memory Optimization
processor = AutoProcessor.from_pretrained(ModelName)

# Use device_map="auto" to automatically handle GPU loading and CPU offloading.
# Use torch_dtype=torch.float16 for half-precision memory efficiency.
model = AutoModelForImageTextToText.from_pretrained(
    ModelName,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# 2. Load the local image
try:
    local_image = Image.open(LOCAL_IMAGE_PATH).convert("RGB")
except FileNotFoundError:
    print(f"Error: Image file not found at {LOCAL_IMAGE_PATH}")
    exit()

# 3. Correctly format the text prompt with the <image> token
question = "tell me about the image"
prompt_template = f"USER: <image>\n{question}\nASSISTANT:"

# 4. Prepare the inputs
# The processor automatically handles the chat template and tokenizes the prompt.
inputs = processor(
    text=prompt_template, 
    images=local_image, 
    return_tensors="pt"
)

# Move input tensors to the model's primary device (determined by device_map="auto")
# You must move them to where the first layer of the model is mapped.
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 5. Generate the output
# The model will use the GPU for computation and offload non-active layers to CPU as configured.
outputs = model.generate(**inputs, max_new_tokens=200)

# 6. Decode and print the result
# Start decoding after the input tokens to get only the generated response.
response = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("Generated Response:", response.strip())