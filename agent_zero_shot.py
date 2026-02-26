from transformers import pipeline

from langchain_community.llms import HuggingFacePipeline

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain.agents import initialize_agent, AgentType

# Load local model
pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_length=200
)

llm = HuggingFacePipeline(pipeline=pipe)

# Create calculator tool
def calculator(expression: str):
    try:
        return str(eval(expression))
    except:
        return "Invalid expression"

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for solving math problems like 45*6 or 100/4"
    )
]

# Initialize Zero-Shot Agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

print("🤖 Zero-Shot Agent Ready (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    response = agent.run(user_input)

    print("\nAI:", response)
    print("-" * 50)