import numpy as np
import tritonclient.http as httpclient

# Define input data
instruction = "retrieve the relevant passage"
text_snippet = "The quick brown fox jumps over the lazy dog."

triton_client = httpclient.InferenceServerClient(
    url="375a-34-125-12-4.ngrok-free.app",
    ssl=True
)

instruction_np = np.array([instruction], dtype=object)
text_snippet_np = np.array([text_snippet], dtype=object)

inputs = [
    httpclient.InferInput("instruction", [1], "BYTES"),
    httpclient.InferInput("text_snippet", [1], "BYTES")
]

inputs[0].set_data_from_numpy(instruction_np)
inputs[1].set_data_from_numpy(text_snippet_np)

try:
    result = triton_client.infer(model_name="e5", inputs=inputs)
    
    embedding = result.as_numpy("embedding")
    
    print("Embedding shape:", embedding.shape)
    print("First 5 values:", embedding[:5])
    print("L2 norm:", np.linalg.norm(embedding))
except Exception as e:
    print("Inference failed:", e)