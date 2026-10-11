# HW2: Deep Personalization - Writeup

**Name:** [Your Name]

**SUNet ID:** [Your ID]


## Setup & Data Collection

### Data Source Description

**TODO: Describe the data source you use for personalizing the model to your own writing style**

## Get Baseline Results (10 points total)

Make sure you have implement `TinkerSampler` in `tinker_personalization.ipynb`. You don't need to answer anything here.


## Method 1: Prompt Engineering (10 points total)

### System Prompt Design (5 points)
Copy-paste the system prompt you engineered to personalize the model outputs.

```text
**TODO: Add the system prompt here.**
```

### Analyze the Pros & Cons of Personalization with Prompting. (5 points)

**TODO: Add your analysis here**


## Method 2: SFT with Synthetic Data (20 points total)

Make sure you have synthesized the input prompts in `tinker_personalization.ipynb` (5 points).

### Launching SFT run

Make sure you have completed `scripts/sft.py`. **Add the wandb link for your run here (10 points):**

(Make sure you have set the project public. Check out [this post](https://community.wandb.ai/t/no-public-option-in-project-visibility-scope/7215/2) if have problem doing so.)

TODO: https://wandb.ai/.../runs/... (replace with your actual link)

### Analyze the Pros & Cons of Personalization with SFT w/ Synthetic Data. (5 points)

**TODO: Add your analysis here**

## Method 3: Reinforcement Learning (RLAIF) (40 points in total)

### Create a Reward Function using LLM-as-a-Judge

#### Train the reward model

Make sure you have complete `rubric_preference_env.py` (5 points).

Add the Tinker checkpoint of the reward model here (5 points):

**TODO: tinker://...**

#### Design and validate your rubric

Copy-paste your rubric. (5 points)

```text
**TODO: Add the rubric here.**
```

### Launching RL run

Make sure you have completed the evaluator in `scripts/rl_with_rubric_rm.py`. **Add the wandb link for your run here (10 points):**

(Make sure you have set the project public. Check out [this post](https://community.wandb.ai/t/no-public-option-in-project-visibility-scope/7215/2) if have problem doing so.)

TODO: https://wandb.ai/.../runs/... (replace with your actual link)


### Analyze the Pros & Cons of Personalization with RL. (15 points)

Make sure your response covers:
1. Reward hacking analysis (5 points)
2. Approach you tried to improve the results (5 points)
3. Inherent limitations you have identified (5 points)

**TODO: Add your analysis here**

## Method 4: DITTO (20 points in total)

### Implementation

**TODO: Add a description of your implementation and choice of hyperparameters.**

### Launching DITTO run

Make sure you have completed `scripts/ditto_dpo.py`. **Add the wandb link for your run here (10 points):**

(Make sure you have set the project public. Check out [this post](https://community.wandb.ai/t/no-public-option-in-project-visibility-scope/7215/2) if have problem doing so.)

TODO: https://wandb.ai/.../runs/... (replace with your actual link)


### Analyze the Pros & Cons of Personalization with DITTO. (10 points)

**TODO: Add your analysis here**
