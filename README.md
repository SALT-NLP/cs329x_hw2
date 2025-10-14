# CS329X HW2: Deep Personalization with Tinker

**Due: Thursday, Oct 30, 2025 at 11:59 PM PT**

100 points total + 40 points extra credit

## Overview

In this assignment, you will experiment with three different approaches to personalize an LLM to generate emails in your style:
1. **Prompt Engineering** - Using system prompts to guide model behavior
2. **Supervised Fine-tuning (SFT) with Synthetic Data** - Training on personalized examples
3. **Reinforcement Learning from AI Feedback (RLAIF)** - Using a reward model to optimize for personalized rubric

We also set two bonuses:
1. Coming up with your own task rather than the default (i.e., email generation). (10 points)
2. Experimenting with [Demonstration-Iterated Task Optimization (DITTO)](https://arxiv.org/pdf/2406.00888). (30 points)

You will use **Tinker**, the cutting-edge training API developed by Thinking Machine Labs, for this assignment. Tinker handles the heavy computation for forward and backward passes, allowing you to complete the assignment on your laptop despite the model training requirements.


## File Structure

### Core Files

- **`tinker_personalization.ipynb`** - Main Jupyter notebook containing the complete assignment workflow. This is where you'll implement most of your code and run experiments.
- **`writeup.md`** - Template for your written responses and analysis. You must fill this out with your answers.

### Training Scripts

- **`scripts/sft.py`** - Script for supervised fine-tuning using synthetic personalized data
- **`scripts/train_rubric_rm.py`** - Script for training a reward model using the Prometheus evaluation framework
- **`scripts/rl_with_rubric_rm.py`** - Script for reinforcement learning training using your trained reward model

### Supporting Modules

- **`rubric_preference_env.py`** - Environment setup and utilities for Prometheus-based evaluation
- **`rubric_preference_types.py`** - Type definitions and data structures for Prometheus evaluation

## Environment Setup

1. Install required packages (we suggest you install them in [Conda environment](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)):
   ```bash
   pip install tinker
   pip install git+https://github.com/thinking-machines-lab/tinker-cookbook.git
   ```

2. Copy `.env.example` to `.env` and fill in the API keys.

3. Open and run through `tinker_personalization.ipynb` following the instructions

## Submission Requirements

### What to Submit

You must submit a **ZIP file** containing:

1. **All code files** including:
   - `tinker_personalization.ipynb` (with all cells executed and outputs visible)
   - `scripts/sft.py`
   - `scripts/train_rubric_rm.py` 
   - `scripts/rl_with_rubric_rm.py`
   - `rubric_preference_env.py`
   - `rubric_preference_types.py`

2. **Completed writeup**:
   - Export `writeup.md` with all TODO sections filled out as a PDF file. 

## Important Notes

- Make sure all wandb projects are set to **public visibility** for grading
- Test that your code runs before submission
- Include all outputs in your Jupyter notebook submission
- Double-check that all TODO sections in `writeup.md` are completed

## Support

If you encounter issues:
1. Check the [Tinker Cookbook](https://tinker-docs.thinkingmachines.ai/) documentation
2. Consult course materials and office hours

Good luck with your personalization experiments!
