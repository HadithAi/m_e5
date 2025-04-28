import torch
from sentence_transformers import SentenceTransformer
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        self.model = SentenceTransformer("BAAI/bge-m3")
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")

    def execute(self, requests):
        responses = []
        for request in requests:
            text_snippet = pb_utils.get_input_tensor_by_name(
                request, "text_snippet"
            ).as_numpy()
            input_sentences = [x.decode("utf-8") for x in text_snippet]

            embeddings = self.model.encode(input_sentences, convert_to_tensor=True)
            embeddings_np = embeddings.cpu().numpy()

            output_tensor = pb_utils.Tensor("embedding", embeddings_np)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses
