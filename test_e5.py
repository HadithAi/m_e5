import numpy as np
import tritonclient.http as httpclient

instruction   = "retrieve the relevant passage"
text_snippet  = "The quick brown fox jumps over the lazy dog."

# note: no "https://" here, and ssl=True
triton_client = httpclient.InferenceServerClient(
    url="f215-34-145-40-185.ngrok-free.app",
    ssl=True
)

# make 2‑D object arrays of shape (1,1)
instruction_np   = np.array([[instruction]],   dtype=object)
text_snippet_np  = np.array([[text_snippet]],  dtype=object)

# declare both inputs with two dims: [batch, seq_count]
inputs = [
    httpclient.InferInput("instruction",   [1, 1], "BYTES"),
    httpclient.InferInput("text_snippet",  [1, 1], "BYTES"),
]
inputs[0].set_data_from_numpy(instruction_np)
inputs[1].set_data_from_numpy(text_snippet_np)

try:
    result    = triton_client.infer(model_name="e5", inputs=inputs)
    embedding = result.as_numpy("embedding")  # should be shape [1, D]
    
    print("Embedding shape:", embedding.shape)
    print("First 5 values:", embedding[0][:5])
    print("L2 norm:", np.linalg.norm(embedding[0]))
except Exception as e:
    print("Inference failed:", e)
