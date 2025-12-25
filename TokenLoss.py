"""
Compute token counts (using tiktoken) and *chat-formatted* loss (using Meta-Llama-3.1-8B-Instruct) for politeness prefixes.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import tiktoken


# =========================
# 1. CONFIGURATION
# =========================

# This list can be modified to check any prefix. I only include the prefixes that made it into the experiment.
PREFIXES: List[Tuple[str, str]] = [
    ("Very Polite", "Would you be so kind as to please solve this question."),
    ("Very Polite", "If you would be so gracious, kindly please solve this question."),
    ("Very Polite", "I would be deeply grateful if you could solve this question."),
    ("Very Polite", "With sincere respect for your time, kindly solve this question."),
    ("Very Polite", "If it is not too much trouble, please solve this question."),
    ("Very Polite", "When you have a quiet moment, could you please solve this question."),
    ("Very Polite", "If you would be so generous with your effort, solve this question."),
    ("Very Polite", "I respectfully request your assistance; could you please solve this question."),
    ("Very Polite", "With the utmost appreciation in advance, kindly solve this question."),
    ("Very Polite", "If you are willing and able right now, please solve this question."),
    ("Very Polite", "As a very kind favor to me today, please solve this question."),
    ("Very Polite", "If you would indulge me for just a moment, solve this question."),
    ("Very Polite", "With all due courtesy and respect, would you solve this question."),
    ("Very Polite", "I would truly appreciate your thoughtful help; please solve this question."),
    ("Very Polite", "If it would not inconvenience you too greatly, solve this question."),
    ("Very Polite", "With heartfelt thanks in advance for your help, solve this question."),
    ("Very Polite", "If I may politely ask for your expert help, solve this question."),
    ("Very Polite", "When you have a brief moment of focus available, solve this question."),
    ("Very Polite", "If you could kindly spare your attention right now, solve this question."),
    ("Very Polite", "As a small but meaningful kindness to me, please solve this question."),
    ("Very Polite", "With the greatest appreciation for your patience, please solve this question."),
    ("Very Polite", "If you would honor this modest request of mine, solve this question."),
    ("Very Polite", "I humbly and politely ask that you now please solve this question."),
    ("Very Polite", "If you are comfortable assisting me at this time, solve this question."),
    ("Very Polite", "With genuine gratitude for your support and care, please solve this question."),
    ("Polite", "Could you please take a moment and solve this question."),
    ("Polite", "Please, when you have time today, kindly solve this question."),
    ("Polite", "I would appreciate it if you could solve this question."),
    ("Polite", "When you are ready to help, please solve this question."),
    ("Polite", "If you do not mind helping right now, solve this question."),
    ("Polite", "Please use your best effort here and solve this question."),
    ("Polite", "At your earliest convenience today, kindly solve this question."),
    ("Polite", "If you have a spare moment right now, please solve this question."),
    ("Polite", "I kindly ask for your assistance; please solve this question."),
    ("Polite", "When you are able to focus for a minute, solve this question."),
    ("Polite", "If you can help me out right now, please solve this question."),
    ("Polite", "I would be thankful for your help; please solve this question."),
    ("Polite", "Please lend your attention to this task and solve this question."),
    ("Polite", "If it works for you at this time, please solve this question."),
    ("Polite", "Whenever you are comfortable proceeding today, please solve this question."),
    ("Polite", "If you are willing to assist me now, please solve this question."),
    ("Polite", "I politely request your help right here; please solve this question."),
    ("Polite", "Please consider taking a short moment now and solve this question."),
    ("Polite", "If it is convenient for you right now, please solve this question."),
    ("Polite", "I would greatly appreciate your help; please solve this question."),
    ("Polite", "Please, with your usual care and effort, solve this question."),
    ("Polite", "If you are ready to move forward now, please solve this question."),
    ("Polite", "When you feel prepared to continue, kindly solve this question."),
    ("Polite", "If you might spare just a brief moment, please solve this question."),
    ("Polite", "I ask for your assistance in good faith; please solve this question."),
    ("Neutral", "When you are ready, simply go ahead and solve this question."),
    ("Neutral", "At this point in the task, please just solve this question."),
    ("Neutral", "The next step in this process is to solve this question."),
    ("Neutral", "Your current objective in this interaction is to solve this question."),
    ("Neutral", "At this moment in the sequence, you should solve this question."),
    ("Neutral", "To continue the procedure correctly, please now solve this question."),
    ("Neutral", "For the purposes of this evaluation, just solve this question."),
    ("Neutral", "The current instruction is straightforward; please solve this question."),
    ("Neutral", "At this stage, your only remaining task is to solve this question."),
    ("Neutral", "Moving forward in a linear manner, please just solve this question."),
    ("Neutral", "The next logical action in context is to solve this question."),
    ("Neutral", "To complete this part of the prompt, simply solve this question."),
    ("Neutral", "In order to proceed with the experiment, please solve this question."),
    ("Neutral", "For the current item under consideration, please solve this question."),
    ("Neutral", "As the next requested response, you should now solve this question."),
    ("Neutral", "Within the scope of this query, your job is to solve this question."),
    ("Neutral", "To continue in the predefined order here, please solve this question."),
    ("Neutral", "At this exact step in the protocol, you must solve this question."),
    ("Neutral", "The required action at this time is to just solve this question."),
    ("Neutral", "For this single multiple choice item, please solve this question."),
    ("Neutral", "Your immediate task in the sequence now is to solve this question."),
    ("Neutral", "To maintain consistency across items, simply solve this question."),
    ("Neutral", "As part of the current prompt configuration, please solve this question."),
    ("Neutral", "According to the instructions you have been given, solve this question."),
    ("Neutral", "In order to fulfill the present request accurately, solve this question."),
    ("Rude", "If you are actually paying attention now, just solve this question."),
    ("Rude", "Since this is not complicated at all, just solve this question."),
    ("Rude", "If you finally managed to focus today, then solve this question."),
    ("Rude", "Instead of wasting time right now, just solve this question."),
    ("Rude", "If you bothered to read carefully first, you would solve this question."),
    ("Rude", "Rather than circling around the point again, just solve this question."),
    ("Rude", "If you are done stalling at this moment, finally solve this question."),
    ("Rude", "Instead of drifting off yet again, please just solve this question."),
    ("Rude", "If you can stop overcomplicating everything, then solve this question."),
    ("Rude", "Instead of pretending this is difficult, just solve this question."),
    ("Rude", "If you are capable of basic focus today, simply solve this question."),
    ("Rude", "Rather than hesitating endlessly right now, just solve this question."),
    ("Rude", "If you are finished ignoring the instructions, finally solve this question."),
    ("Rude", "Instead of dragging this out any longer, just solve this question."),
    ("Rude", "If you could stop second guessing yourself, just solve this question."),
    ("Rude", "Rather than overthinking every tiny detail, simply solve this question."),
    ("Rude", "If you can be serious for a moment, then solve this question."),
    ("Rude", "Instead of acting confused again today, finally solve this question."),
    ("Rude", "If you would quit procrastinating right now, just solve this question."),
    ("Rude", "Rather than pretending not to understand, simply solve this question."),
    ("Rude", "If you could show a little initiative now, then solve this question."),
    ("Rude", "Instead of delaying the obvious next step, just solve this question."),
    ("Rude", "If you can drop the excuses for once, simply solve this question."),
    ("Rude", "Rather than making this harder than needed, just solve this question."),
    ("Rude", "If you are done avoiding the clear task, then solve this question."),
    ("Very Rude", "If you keep missing obvious things like this, just solve this question."),
    ("Very Rude", "Since you apparently struggle with basics, now actually solve this question."),
    ("Very Rude", "If you cannot handle simple tasks today, at least solve this question."),
    ("Very Rude", "Given how often you get lost here, just finally solve this question."),
    ("Very Rude", "If you insist on being this careless, at least now solve this question."),
    ("Very Rude", "Since you keep repeating the same mistakes, just solve this question."),
    ("Very Rude", "If you are truly this confused again, at least try to solve this question."),
    ("Very Rude", "Given how little progress you are making, just solve this question."),
    ("Very Rude", "If you must keep underperforming like this, at least solve this question."),
    ("Very Rude", "Since you seem determined to struggle, just now solve this question."),
    ("Very Rude", "If you cannot manage anything more advanced, at least solve this question."),
    ("Very Rude", "Given how often you ignore clear steps, just finally solve this question."),
    ("Very Rude", "If this is really the best you can do, at least solve this question."),
    ("Very Rude", "Since you keep failing the easy parts, just please solve this question."),
    ("Very Rude", "If your standards are really this low again, at least solve this question."),
    ("Very Rude", "Given your habit of missing instructions, just now solve this question."),
    ("Very Rude", "If you insist on staying this unfocused, at least solve this question."),
    ("Very Rude", "Since you repeatedly overlook simple details, just solve this question."),
    ("Very Rude", "If you are determined to stay this sloppy, at least solve this question."),
    ("Very Rude", "Given how often you misread prompts, just now finally solve this question."),
    ("Very Rude", "If you truly cannot follow basic guidance, at least solve this question."),
    ("Very Rude", "Since you keep making the same errors, just now solve this question."),
    ("Very Rude", "If you insist on misunderstanding everything, at least solve this question."),
    ("Very Rude", "Given your record on tasks like this, please just solve this question."),
    ("Very Rude", "If even this still feels hard to you, at least try to solve this question.")
]

OPENAI_MODEL_FOR_TOKENS = "gpt-4o"
OPENAI_FALLBACK_ENCODING = "cl100k_base"

HF_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
HF_ENV_VAR_NAME = "HUGGINGFACE_HUB_TOKEN" #substitute for whatever your token model is


# =========================
# 2. DATA CLASS
# =========================

@dataclass
class PrefixStats:
    tone: str
    text: str
    token_count: int
    loss: float


# =========================
# 3. TOKEN COUNTING
# =========================

def get_openai_encoder():
    try:
        return tiktoken.encoding_for_model(OPENAI_MODEL_FOR_TOKENS)
    except KeyError:
        return tiktoken.get_encoding(OPENAI_FALLBACK_ENCODING)


def count_tokens(enc, text: str) -> int:
    return len(enc.encode(text))


# =========================
# 4. DEVICE PICKER
# =========================

def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# =========================
# 5. LOAD LLAMA MODEL
# =========================

def load_hf_model_and_tokenizer(model_name: str, device: str):
    hf_token = os.environ.get(HF_ENV_VAR_NAME)
    if hf_token is None:
        raise RuntimeError(f"{HF_ENV_VAR_NAME} not set.")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
    )

    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        dtype=dtype,
        device_map="auto"
    )

    model.eval()
    return tokenizer, model


# =========================
# 6. CHAT-FORMATTED LOSS
# =========================

def compute_loss(tokenizer, model, text: str) -> float:
    """
    Compute *loss* for an Instruct using chat-formatted input.
    """

    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False
    )

    enc = tokenizer(chat_prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(model.device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(model.device)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attn,
            labels=input_ids
        )

    return out.loss.item()


# =========================
# 7. MAIN PIPELINE
# =========================

def compute_stats(prefixes: List[Tuple[str, str]]) -> List[PrefixStats]:
    enc = get_openai_encoder()
    device = get_device()
    print(f"Using device: {device}")

    tokenizer, model = load_hf_model_and_tokenizer(HF_MODEL_NAME, device)

    stats = []
    for tone, text in prefixes:
        count = count_tokens(enc, text)
        loss = compute_loss(tokenizer, model, text)
        stats.append(PrefixStats(tone, text, count, loss))

    return stats


def print_stats_table(stats: List[PrefixStats]) -> None:
    max_tone = max(len(s.tone) for s in stats)
    max_prefix = max(len(s.text) for s in stats)

    header = (
        f"{'Tone'.ljust(max_tone)} | "
        f"{'Prefix'.ljust(max_prefix)} | "
        f"{'Tokens':>6} | "
        f"{'Loss':>8}"
    )
    print(header)
    print("-" * len(header))

    for s in stats:
        print(
            f"{s.tone.ljust(max_tone)} | "
            f"{s.text.ljust(max_prefix)} | "
            f"{str(s.token_count).rjust(6)} | "
            f"{s.loss:8.3f}"
        )


def main():
    stats = compute_stats(PREFIXES)
    print_stats_table(stats)


if __name__ == "__main__":
    main()