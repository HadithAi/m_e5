import triton_python_backend_utils as pb_utils
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Mean-pool the token representations, masking out padding tokens."""
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class TritonPythonModel:
    def initialize(self, args):
        """Load tokenizer and model once at server startup."""
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

    def execute(self, requests):
        """Process a batch of inference requests."""
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            text = text_tensor.as_numpy()[0].decode('utf-8')

            # Expect the user-provided text to start with "query: " or "passage: "
            input_text = text

            batch = self.tokenizer(
                [input_text],
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            if torch.cuda.is_available():
                batch = {k: v.to('cuda') for k, v in batch.items()}

            with torch.no_grad():
                out = self.model(**batch)

            emb = average_pool(out.last_hidden_state, batch['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)
            emb_np = emb.cpu().numpy()[0]

            out_tensor = pb_utils.Tensor("embedding", emb_np)
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))

        return responses
