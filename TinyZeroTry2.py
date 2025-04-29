import argparse
import os

# === TinyZero GSM8K Integration with Fine-Tuning and Early Stopping ===
import random
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

# === Sets up arguments for the run script ===
def arg_setup():
    parser = argparse.ArgumentParser(
        prog="ecen743-tinyzero-trainer"
    )

    parser.add_argument('-m', '--model', type=str, required=True, help='Model to be used [tinyzero, gsm_trained_tinyzero]')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='The dataset to be used [gsm8k, prm800k]')
    parser.add_argument('-p', '--problems', type=int, required=True, help='Number of problems to be used to train [0-5000]')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output directory for the model and testing loss csv')
    return parser.parse_args()

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
# values that works with example problem: 1e-6, 2, 0.0001
def gsm8k_config():
    return {
        "lr": 1e-8,      # Initial value: 1e-5, 
        "patience": 15,           # Number of epochs without improvement before stopping
        "min_delta": 0.0001       # Minimum change in loss to qualify as improvement
        # "accuracy_req_steps": 3,    # Minimum number of times the loss needs to be less than or equal to "accuracy_req"
        # "accuracy_req": 0.15        # Adequate loss for training.
    }

# === GSM8K problem sample ===

from datasets import load_dataset

def load_gsm8k_split(split="train", num_examples=10):
    dataset = load_dataset("gsm8k", "main", split=split)
    problems = []

    for ex in dataset.select(range(num_examples)):
        question = ex["question"].strip()
        answer = ex["answer"].strip()

        # Split steps from answer (which is a string)
        steps = answer.split("\n")
        steps = [s.strip() for s in steps if s.strip()]
        problems.append({"question": question, "steps": steps})

    return problems


# def get_sample_problem():
#     return {
#         "question": "Tom had 3 apples. He buys 2 more. How many apples does he have?",
#         "steps": [
#             "Tom starts with 3 apples.",
#             "He buys 2 more.",
#             "3 + 2 = 5",
#             "Answer: 5 apples."
#         ]
#     }

def fine_tune_llm(model, tokenizer, device, train_problems, test_problems, config):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    best_loss = float("inf")
    patience_counter = 0
    epoch = 0
    loss_overall = []
    training_losses = []
    while True:
        total_loss = 0.0
        train_prob_num = 0
        for index, problem in enumerate(train_problems):
            # print(f"Training Problem {index+1}")
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
            train_prob_num = index

        epoch += 1
        training_loss = total_loss/train_prob_num
        print(f"Epoch {epoch}: Training Loss = {training_loss:.4f}")
        training_losses.append(training_loss)

        # === Evaluate on test set ===
        model.eval()
        test_loss = 0.0
        test_prob_num = 0
        with torch.no_grad():
            for test_index, problem in enumerate(test_problems):
                # print(f"Testing problem {test_index+1}")
                state = problem["question"]
                for gold_step in problem["steps"]:
                    prompt = state + "\nStep:"
                    input_text = prompt + gold_step
                    encodings = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
                    input_ids = encodings["input_ids"]
                    labels = input_ids.clone()
                    outputs = model(input_ids=input_ids, labels=labels)
                    test_loss += outputs.loss.item()
                    state += "\n" + gold_step
                test_prob_num = test_index+1

        print(f" Test Loss = {test_loss/test_prob_num:.4f}")
        loss_overall.append(test_loss/test_prob_num)

        if best_loss - test_loss > config["min_delta"]:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print("Early stopping triggered.")
                break
    return loss_overall

# def fine_tune_llm(model, tokenizer, device, problems, config):
#     model.train()
#     optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

#     best_loss = float("inf")
#     patience_counter = 0
#     epoch = 0

#     while True:
#         total_loss = 0.0
#         for index, problem in enumerate(problems):
#             print(f"For problem {index+1}")
#             game = GSM8KGame(problem["question"], problem["steps"])
#             state = game.make_image()

#             for gold_step in problem["steps"]:
#                 prompt = state + "\nStep:"
#                 input_text = prompt + gold_step
#                 encodings = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
#                 input_ids = encodings["input_ids"]
#                 labels = input_ids.clone()

#                 outputs = model(input_ids=input_ids, labels=labels)
#                 loss = outputs.loss
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()

#                 total_loss += loss.item()
#                 state += "\n" + gold_step

#         epoch += 1
#         print(f"Epoch {epoch}: Total Loss = {total_loss:.4f}")

#         # Early stopping check
#         if best_loss - total_loss > config["min_delta"]:
#             best_loss = total_loss
#             patience_counter = 0
#         else:
#             patience_counter += 1
#             if patience_counter >= config["patience"]:
#                 print("Early stopping triggered.")
#                 break

# === Evaluate model response to a problem ===
def evaluate_response(model, tokenizer, device, problem):
    model.eval()
    state = problem["question"]
    print("\n=== Model Response After Fine-Tuning ===")
    print(f"Question: {state}\n")

    for _ in range(len(problem["steps"])):
        prompt = state + "\nStep:"
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_k=40
        )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated[len(prompt):].strip().split("\n")[0]
        print(f"Step: {response}")
        state += "\n" + response

    print("\n--- Ground Truth ---")
    for step in problem["steps"]:
        print(f"Step: {step}")

# === Main fine-tuning loop ===
def main():
    args = arg_setup()
    config = gsm8k_config()
    
    if args.model == 'tinyzero':
        # model_name = "rayliuray/TinyZero-CountDown-Qwen2.5-3b-GRPO-Step10" # too large for my GPU
        model_name = "rayliuray/TinyZero-CountDown-Qwen2.5-3b-GRPO-Step10" # too large for my GPU
    elif args.model == 'gsm_trained_tinyzero':
        model_name = "./gsm_trained_tinyzero" # too large for my GPU
    elif args.model == "tinyzero-1.5":
        model_name = "roastduckkiller/TinyZero-DO"
    elif args.model == "gsm_trained_tinyzero-1.5":
        model_name = "./gsm_trained_tinyzero-1.5"
    else:
        print("Please provide exising model name! [tinyzero, gsm_trained_tinyzero, tinyzero-1.5]")
    # model_name = "./tinyzero_GSM_trained" # This is the directory to the GSM trained TinyZero, uncomment this to start tuning for PSM
    # model_name = "roastduckkiller/TinyZero-DO"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    print("Loading in Problems")
    def load_gsm8k_split(total=100, test_ratio=0.2):
        dataset = load_dataset("gsm8k", "main", split="train")
        dataset = dataset.shuffle(seed=42).select(range(total))
        problems = []

        for ex in dataset:
            question = ex["question"].strip()
            answer = ex["answer"].strip()
            steps = answer.split("\n")
            steps = [s.strip() for s in steps if s.strip()]
            problems.append({"question": question, "steps": steps})

        split_idx = int(len(problems) * (1 - test_ratio))
        return problems[:split_idx], problems[split_idx:]

    def load_prm800k_split(total=100, test_ratio=0.2):
        dataset = load_dataset("RyanYr/MATH-prm800k", split="train")
        dataset = dataset.shuffle(seed=42).select(range(total))
        problems = []

        for ex in dataset:
            question = ex["problem"].strip()
            answer = ex["solution"].strip()
            steps = answer.split(".")
            steps = [s.strip() for s in steps if s.strip()]
            problems.append({"question": question, "steps": steps})

        split_idx = int(len(problems) * (1 - test_ratio))
        return problems[:split_idx], problems[split_idx:]

    total_problems = args.problems
    if args.dataset == 'gsm8k':
        train_problems, test_problems = load_gsm8k_split(total=total_problems, test_ratio=0.2)
    elif args.dataset == 'prm800k':
        train_problems, test_problems = load_gsm8k_split(total=total_problems, test_ratio=0.2)
    else:
        print("Please provide a valid dataset [gsm8k, prm800k]")
        return
    test_problem = test_problems[0]

    print("Creating output directory")
    os.system(f"mkdir -p ./{args.output}")

    print("Starting to train the model")
    loss = fine_tune_llm(model, tokenizer, device, train_problems, test_problems, config)
    pd.DataFrame(loss).to_csv(f"./{args.output}/TestingLoss_{total_problems}.csv",index=False,header=False)
    evaluate_response(model, tokenizer, device, test_problem)

    model.save_pretrained(f"./{args.output}")
    tokenizer.save_pretrained(f"./{args.output}")

    print(f"Model trained and saved in {args.output}")

if __name__ == "__main__":
    main()