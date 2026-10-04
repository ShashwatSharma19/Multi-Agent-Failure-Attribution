"""Prompt builders copied verbatim (in wording) from the reference implementation.

The reference includes the ground-truth answer inside the prompt (its own comment
flags this as optional). We keep ``include_ground_truth=True`` by default to
reproduce the published numbers, but expose the flag so Phase 3+ can run the
honest, answer-free ablation.
"""

from __future__ import annotations

ALL_AT_ONCE_SYSTEM = "You are a helpful assistant skilled in analyzing conversations."
STEP_SYSTEM = "You are a precise step-by-step conversation evaluator."
BINARY_SYSTEM = "You are an AI assistant specializing in localizing errors in conversation segments."

# Shown inside the all-at-once prompt as an illustration of step numbering.
_STEP_EXAMPLE = """
            {
                "agent a": "xx",
                "agent b": "xxxx",
                "agent c": "xxxxx",
                "agent a": "xxxxxxx"
            },
            """


def _answer_line(ground_truth: str, include: bool) -> str:
    return f"The Answer for the problem is: {ground_truth}\n" if include else ""


def all_at_once_prompt(problem: str, ground_truth: str, transcript: str, include_ground_truth: bool = True) -> str:
    return (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {problem}\n"
        + _answer_line(ground_truth, include_ground_truth)
        + "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n" + transcript
        + "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        + _STEP_EXAMPLE
        + "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
    )


def step_prompt(problem: str, ground_truth: str, history_so_far: str, idx: int, agent_name: str, include_ground_truth: bool = True) -> str:
    return (
        f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
        + _answer_line(ground_truth, include_ground_truth)
        + f"Here is the conversation history up to the current step:\n{history_so_far}\n"
        f"The most recent step ({idx}) was by '{agent_name}'.\n"
        "Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
        "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
        "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
        "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
    )


def binary_prompt(problem: str, ground_truth: str, segment: str, range_desc: str, upper_desc: str, lower_desc: str, include_ground_truth: bool = True) -> str:
    return (
        "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
        "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user’s query.\n"
        f"The problem to address is as follows: {problem}\n"
        + _answer_line(ground_truth, include_ground_truth)
        + f"Review the following conversation segment {range_desc}:\n\n{segment}\n\n"
        f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_desc}) or the lower half ({lower_desc}) of this segment.\n"
        "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that."
    )
