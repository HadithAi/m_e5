import tritonclient.http as httpclient
import numpy as np

triton_client = httpclient.InferenceServerClient(url="localhost:8000")
instruction = np.array(["Given a web search query, retrieve relevant passages that answer the query"], dtype=object)
text_snippet = np.array(["how much protein should a female eat"], dtype=object)

inputs = [
    httpclient.InferInput("instruction", [1], "STRING").set_data_from_numpy(instruction),
    httpclient.InferInput("text_snippet", [1], "STRING").set_data_from_numpy(text_snippet)
]
outputs = [httpclient.InferRequestedOutput("embedding")]

response = triton_client.infer(model_name="embedding_model", inputs=inputs, outputs=outputs)
embedding = response.as_numpy("embedding")  # Shape: [1024]
print(embedding)