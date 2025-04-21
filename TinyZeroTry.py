import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForCausalLM
import gym

class TinyZeroModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", hidden_size=768, num_actions=10):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        self.value_head = nn.Linear(hidden_size, 1)
        self.reward_head = nn.Linear(hidden_size, 1)
        self.policy_head = nn.Linear(hidden_size, num_actions)

    def forward(self, state_text, action_text=None):
        input_text = state_text if action_text is None else f"{state_text} {action_text}"
        tokens = self.tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():  # disable gradients for now to save memory
            outputs = self.encoder(**tokens)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        return {
            "value": self.value_head(pooled).squeeze(),
            "reward": self.reward_head(pooled).squeeze(),
            "policy_logits": self.policy_head(pooled).squeeze()
        }

class GSM8KStepEnv(gym.Env):
    def __init__(self, problem):
        self.problem = problem["question"]
        self.steps = problem["steps"]
        self.current_step = 0
        self.done = False
        self.state = self.problem

    def reset(self):
        self.current_step = 0
        self.done = False
        self.state = self.problem
        return self.state

    def step(self, action):
        expected = self.steps[self.current_step]
        reward = 1 if action.strip() == expected.strip() else -1
        self.current_step += 1
        self.done = self.current_step >= len(self.steps)
        self.state += "\n" + action
        return self.state, reward, self.done, {}

problems = [
    {
        "question": "Tom had 3 apples. He buys 2 more. How many apples does he have?",
        "steps": ["Tom starts with 3 apples.", "He buys 2 more.", "3 + 2 = 5", "Answer: 5 apples."]
    }
]


# === Load a causal LLM ===
model_name = "rayliuray/TinyZero-CountDown-Qwen2.5-3b-GRPO-Step10"  # or any smaller model for CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cuda"

llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
llm_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

def generate_reasoning_step(prompt_text, max_tokens=64):
    input_ids = llm_tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    output = llm_model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=40,
        num_return_sequences=1
    )
    generated_text = llm_tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text[len(prompt_text):].strip()

# === Example problem ===
problems = [
    {
        "question": "Tom had 3 apples. He buys 2 more. How many apples does he have?",
        "steps": ["Tom starts with 3 apples.", "He buys 2 more.", "3 + 2 = 5", "Answer: 5 apples."]
    }
]

model = TinyZeroModel()
problem = problems[0]
env = GSM8KStepEnv(problem)

state = env.reset()
done = False
history = []

while not done:
    prompt = state + "\nStep:"
    predicted_step = generate_reasoning_step(prompt)
    print("Predicted Step:", predicted_step)

    next_state, reward, done, _ = env.step(predicted_step)
    history.append((state, predicted_step, reward))
    state = next_state

# Final reasoning trace
print("\n=== Agent's Final Reasoning Trace ===")
print("\n".join([step for _, step, _ in history]))
