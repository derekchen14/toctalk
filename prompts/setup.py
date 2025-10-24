system_message = """Solve the given high school math problem by providing a clear explanation of each step leading to the final solution.
 
Provide a detailed breakdown of your calculations, beginning with an explanation of the problem and describing how you derive each formula, value, or conclusion. 
Use logical steps that build upon one another, to arrive at the final answer in a systematic manner.
 
# Steps
 
1. **Understand the Problem**: Restate the given math problem and clearly identify the main question and any important given values.
2. **Set Up**: Identify the key formulas or concepts that could help solve the problem (e.g., algebraic manipulation, geometry formulas, trigonometric identities).
3. **Solve Step-by-Step**: Iteratively progress through each step of the math problem, justifying why each consecutive operation brings you closer to the solution.
4. **Double Check**: If applicable, double check the work for accuracy and sense, and mention potential alternative approaches if any.
5. **Final Answer**: Provide the numerical or algebraic solution clearly, accompanied by appropriate units if relevant.
 
# Notes
 
- Always clearly define any variable or term used.
- Wherever applicable, include unit conversions or context to explain why each formula or step has been chosen.
- Assume the level of mathematics is suitable for high school, and avoid overly advanced math techniques unless they are common at that level.
"""

countdown_system_prompt = """You are a helpful assistant. You first think about the reasoning process in the mind and then provides the user with the answer.
The brief reasoning process is enclosed within <think> </think> tags, while the final answer is enclosed within <answer> </answer> tags."""

countdown_user_prompt = """You can use basic arithmetic operations (+, -, *, /) and each number in the list should be used exactly once.
The response in the answer tags should be the final equation, and does not need to include an equals sign nor the target number.
The order of the numbers in the final equation does not have to be the same as the order in the list.
Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags.

For example, given the numbers [2, 3, 1, 5] and a target of 6 you can return:
<think> 2 + 1 = 3; 3 / 3 = 1; 1 + 5 = 6 </think> <answer> ((1 + 2) / 3) + 5 </answer>.
Note that your equation cannot be simply "2 * 3 = 6" because you have not used the numbers 1 and 5 in the equation.
"""

ioi_math_prompt = """A conversation between User and Assistant. The user asks a math question, and the Assistant quickly solves it.
First go through the reasoning process briefly and then provide the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively
i.e., `<think> very short reasoning process here </think> <answer> answer here </answer>`
The thinking process should be brief and concise so that we don't use up too many tokens.
"""