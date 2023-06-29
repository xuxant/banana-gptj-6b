from transformers import GPTJForCausalLM, GPT2Tokenizer
import torch


def download_model():
    GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")


if __name__ == "__main__":
    download_model()