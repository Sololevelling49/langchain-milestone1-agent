from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# Load model
pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_length=200
)

llm = HuggingFacePipeline(pipeline=pipe)

# Create prompt
prompt = PromptTemplate.from_template(
    """
You are a professional AI assistant.
Answer clearly and concisely.

Question: {question}
Answer:
"""
)

# LCEL Chain (Modern Way)
chain = prompt | llm

# Run
response = chain.invoke({"question": "What is an AI agent?"})

print("\nResponse:\n")
print(response)