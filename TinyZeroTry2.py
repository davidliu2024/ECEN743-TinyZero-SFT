# === TinyZero GSM8K Integration with Fine-Tuning ===
import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

# === GSM8K-style environment class ===
class GSM8KGame:
    def __init__(self, question, steps):
        self.question = question
        self.steps = steps
        self.current_step = 0
        self.done = False
        self.history = [question]
        self.total_reward = 0

    def legal_actions(self):
        return list(range(len(self.steps)))

    def terminal(self):
        return self.done

    def clone(self):
        cloned = GSM8KGame(self.question, self.steps)
        cloned.current_step = self.current_step
        cloned.done = self.done
        cloned.history = list(self.history)
        cloned.total_reward = self.total_reward
        return cloned

    def apply(self, action):
        expected_step = self.steps[self.current_step]
        reward = 1 if action.strip() == expected_step.strip() else -1
        self.total_reward += reward
        self.history.append(action)
        self.current_step += 1
        if self.current_step >= len(self.steps):
            self.done = True
        return reward

    def make_image(self):
        return " ".join(self.history)

# === TinyZero configuration for GSM8K ===
def gsm8k_config():
    return {
        "lr": 1e-5,
        "epochs": 3
    }

# === GSM8K problem sample ===
def get_sample_problem():
    return {
        "question": "Tom had 3 apples. He buys 2 more. How many apples does he have?",
        "steps": [
            "Tom starts with 3 apples.",
            "He buys 2 more.",
            "3 + 2 = 5",
            "Answer: 5 apples."
        ]
    }

# === Fine-tune LLM using teacher-forced steps ===
def fine_tune_llm(model, tokenizer, device, problems, config):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(config["epochs"]):
        total_loss = 0.0
        for problem in problems:
            game = GSM8KGame(problem["question"], problem["steps"])
            state = game.make_image()

            for gold_step in problem["steps"]:
                prompt = state + "\nStep:"
                input_text = prompt + gold_step
                encodings = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
                input_ids = encodings["input_ids"]
                labels = input_ids.clone()

                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                state += "\n" + gold_step

        print(f"Epoch {epoch + 1}: Total Loss = {total_loss:.4f}")

# === Main fine-tuning loop ===
def main():
    config = gsm8k_config()
    model_name = "rayliuray/TinyZero-CountDown-Qwen2.5-3b-GRPO-Step10"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    problems = [get_sample_problem()]
    fine_tune_llm(model, tokenizer, device, problems, config)

if __name__ == "__main__":
    main()
