from potassium import Potassium, Request, Response
from transformers import GPTJForCausalLM, GPT2Tokenizer
import torch

app = Potassium("diffusion")
device = "cuda:0" if torch.cuda.is_available() else "cpu"

@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    context = {
        "model": model,
        "tokenizer": tokenizer
    }

    return context




@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    tokenizer = context.get("tokenizer")
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    model = context.get("model")
    outputs = model.generate(input_ids)
    result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    return Response(
        json={"output": result},
        status=200
    )


if __name__ == "__main__":
    app.serve()