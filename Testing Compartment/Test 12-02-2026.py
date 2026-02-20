import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/blenderbot-400M-distill"
MAX_TURNS = 6


def main() -> int:
    print("Loading model... (first time may take a few minutes)")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    except Exception as exc:
        print(f"Failed to load model '{MODEL_NAME}': {exc}")
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print("\nBlenderBot is ready! Type 'exit' to quit.\n")

    # Store turns as simple tagged strings so context is clearer to the model.
    chat_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            break

        chat_history.append(f"User: {user_input}")
        context = " </s> ".join(chat_history[-MAX_TURNS:])

        inputs = tokenizer(
            context,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            reply_ids = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=True,
                top_k=40,
                top_p=0.92,
                temperature=0.75,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
            )

        reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()
        if not reply:
            reply = "I'm not sure how to respond to that."

        print(f"Bot: {reply}\n")
        chat_history.append(f"Bot: {reply}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
