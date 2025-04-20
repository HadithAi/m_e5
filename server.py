#!/usr/bin/env python3
import logging
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton, TritonConfig
from pyngrok import ngrok, conf

logger = logging.getLogger("e5_embedding_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct")
model.eval()
if torch.cuda.is_available():
    model = model.to("cuda")
    logger.info("Model moved to GPU")

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

@batch
def _infer_fn(**inputs: np.ndarray):
    """Inference function for batched requests."""
    instruction_batch, text_snippet_batch = inputs.values()

    instruction_batch = np.char.decode(instruction_batch.astype("bytes"), "utf-8")
    text_snippet_batch = np.char.decode(text_snippet_batch.astype("bytes"), "utf-8")

    combined_texts = [
        f"Instruct: {instr.item()}\nQuery: {text.item()}"
        for instr, text in zip(instruction_batch, text_snippet_batch)
    ]

    batch_dict = tokenizer(
        combined_texts,
        max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    if torch.cuda.is_available():
        batch_dict = {k: v.to("cuda") for k, v in batch_dict.items()}

    with torch.no_grad():
        outputs = model(**batch_dict)

    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings_np = embeddings.cpu().numpy()

    return {"embedding": embeddings_np}

NGROK_AUTH_TOKEN = "2vyx4apXECvTFqr9pTU213ErpUv_4d4PL9jTStxyrWquUPSEZ"  
conf.get_default().auth_token = NGROK_AUTH_TOKEN

logger.info("Starting ngrok tunnel for HTTP port 8015")
http_tunnel = ngrok.connect(8015, proto="http", bind_tls=True)
public_url = http_tunnel.public_url
print(f"**************Ngrok tunnel established at: {public_url}")

config = TritonConfig(http_port=8015, grpc_port=8016, metrics_port=8017)
with Triton(config=config) as triton:
    logger.info("Loading multilingual-e5-large-instruct model.")
    triton.bind(
        model_name="e5",
        infer_func=_infer_fn,
        inputs=[
            Tensor(name="instruction", dtype=np.bytes_, shape=(1)),
            Tensor(name="text_snippet", dtype=np.bytes_, shape=(1))
        ],
        outputs=[
            Tensor(name="embedding", dtype=np.float32, shape=(1024))
        ],
        config=ModelConfig(max_batch_size=2),
        strict=True,
    )
    logger.info("Serving inference")
    triton.serve()