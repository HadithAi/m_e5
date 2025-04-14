import numpy as np
import tritonclient.http as httpclient

# Define input data
instruction = "retrieve the relevant passage"
text_snippet = "The quick brown fox jumps over the lazy dog."

# Create Triton client
triton_client = httpclient.InferenceServerClient(url="localhost:8000")

# Prepare input tensors
instruction_np = np.array([instruction], dtype=object)
text_snippet_np = np.array([text_snippet], dtype=object)

# Create InferInput objects
inputs = [
    httpclient.InferInput("instruction", [1], "BYTES"),
    httpclient.InferInput("text_snippet", [1], "BYTES")
]

# Set data for inputs
inputs[0].set_data_from_numpy(instruction_np)
inputs[1].set_data_from_numpy(text_snippet_np)

try:
    # Perform inference
    result = triton_client.infer(model_name="e5", inputs=inputs)
    
    # Get the output embedding
    embedding = result.as_numpy("embedding")
    
    # Print the results
    print("Embedding shape:", embedding.shape)
    print("First 5 values:", embedding[:5])
    print("L2 norm:", np.linalg.norm(embedding))
except Exception as e:
    print("Inference failed:", e)
