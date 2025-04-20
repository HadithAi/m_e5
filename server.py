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
import argparse

logger = logging.getLogger("e5_embedding_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class _E5InferFuncWrapper:
    def __init__(self, model, tokenizer, device):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    @batch
    def __call__(self, **inputs: np.ndarray):
        instruction_batch, text_snippet_batch = inputs.values()

        instruction_batch = np.char.decode(instruction_batch.astype("bytes"), "utf-8")
        text_snippet_batch = np.char.decode(text_snippet_batch.astype("bytes"), "utf-8")

        combined_texts = [
            f"Instruct: {instr.item()}\nQuery: {text.item()}"
            for instr, text in zip(instruction_batch, text_snippet_batch)
        ]

        batch_dict = self._tokenizer(
            combined_texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        batch_dict = {k: v.to(self._device) for k, v in batch_dict.items()}

        with torch.no_grad():
            outputs = self._model(**batch_dict)

        embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_np = embeddings.cpu().numpy()

        return {"embedding": embeddings_np}

def _infer_function_factory(devices):
    infer_funcs = []
    for device in devices:
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large-instruct")
        model = AutoModel.from_pretrained("intfloat/multilingual-e5-large-instruct")
        model.eval()
        model = model.to(device)
        infer_funcs.append(_E5InferFuncWrapper(model, tokenizer, device))
    return infer_funcs

NGROK_AUTH_TOKEN = "2vyx4apXECvTFqr9pTU213ErpUv_4d4PL9jTStxyrWquUPSEZ"  
conf.get_default().auth_token = NGROK_AUTH_TOKEN

logger.info("Starting ngrok tunnel for HTTP port 8015")
http_tunnel = ngrok.connect(8015, proto="http", bind_tls=True)
public_url = http_tunnel.public_url
print(f"**************Ngrok tunnel established at: {public_url}")

parser = argparse.ArgumentParser()
parser.add_argument("--num-instances", type=int, default=1, help="Number of model instances")
args = parser.parse_args()

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    devices = [f"cuda:{i % num_gpus}" for i in range(args.num_instances)]
else:
    devices = ["cpu"] * args.num_instances

config = TritonConfig(http_port=8015, grpc_port=8016, metrics_port=8017)
with Triton(config=config) as triton:
    logger.info(f"Loading {args.num_instances} instances of multilingual-e5-large-instruct model on devices: {devices}")
    infer_funcs = _infer_function_factory(devices)
    triton.bind(
        model_name="e5",
        infer_func=infer_funcs,
        inputs=[
            Tensor(name="instruction", dtype=np.bytes_, shape=()),
            Tensor(name="text_snippet", dtype=np.bytes_, shape=())
        ],
        outputs=[
            Tensor(name="embedding", dtype=np.float32, shape=())
        ],
        config=ModelConfig(max_batch_size=1),
        strict=True,
    )
    logger.info("Serving inference")
    triton.serve()