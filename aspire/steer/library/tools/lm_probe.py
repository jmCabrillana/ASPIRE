"""Toy probe: does the base SmolVLM2 language model (32 layers + lm_head, general English) know which in-vocabulary word a
paraphrase word means? Scores candidate rewrites by paraphrase log-likelihood in both directions."""
import sys, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForImageTextToText.from_pretrained(NAME, torch_dtype=torch.bfloat16).cuda().eval()
lm = model.model.text_model if hasattr(model.model, "text_model") else model.model
head = model.lm_head

@torch.no_grad()
def logp(prompt, target):
    p = tok(prompt, add_special_tokens=False)["input_ids"]; t = tok(target, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([p + t]).cuda()
    h = lm(input_ids=ids).last_hidden_state
    lp = torch.log_softmax(head(h).float(), -1)[0, len(p) - 1:-1]
    return float(lp.gather(1, torch.tensor(t).cuda()[:, None]).sum()), len(t)

def score(para, cand):
    a, _ = logp(f"Robot command: {cand}\nThe same command in other words: ", para)
    b, n = logp(f"Robot command: {para}\nThe same command in other words: ", cand)
    return a + b

if __name__ == "__main__":
    para = sys.argv[1]; slot = sys.argv[2]; words = sys.argv[3].split(",")
    res = sorted(((score(para, slot.replace("_", w).strip()), w) for w in words), reverse=True)
    for s, w in res[:10]: print(f"{s:8.2f}  {slot.replace('_', w)}")
