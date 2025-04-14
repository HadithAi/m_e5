import triton_python_backend_utils as pb_utils
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class TritonPythonModel:
    def initialize(self, args):
        """Load the tokenizer and model during server startup."""
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large-instruct')
        self.model.eval()
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def execute(self, requests):
        """Process inference requests."""
        responses = []
        for request in requests:
            # Extract inputs
            instr_tensor = pb_utils.get_input_tensor_by_name(request, "instruction")
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text_snippet")
            instruction = instr_tensor.as_numpy()[0].decode('utf-8')  # Single string
            text_snippet = text_tensor.as_numpy()[0].decode('utf-8')  # Single string

            # Combine instruction and text (mimicking get_detailed_instruct)
            combined_text = f'Instruct: {instruction}\nQuery: {text_snippet}'

            # Tokenize
            batch_dict = self.tokenizer(
                [combined_text],  # List with one text
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            if torch.cuda.is_available():
                batch_dict = {k: v.to('cuda') for k, v in batch_dict.items()}

            # Model inference
            with torch.no_grad():
                outputs = self.model(**batch_dict)

            # Compute embedding
            embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embedding_np = embeddings.cpu().numpy()[0]  # Shape: [1024]

            # Create output tensor
            output_tensor = pb_utils.Tensor("embedding", embedding_np)
            response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
            responses.append(response)

        return responses