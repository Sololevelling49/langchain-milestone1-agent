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

# Create prompt template
prompt = PromptTemplate.from_template(
    """
You are a helpful AI assistant.
Provide structured answers.

Question: {question}
Answer:
"""
)

# Modern LCEL chain
chain = prompt | llm

print("🤖 LangChain Console Assistant (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye 👋")
        break

    response = chain.invoke({"question": user_input})

    print("\nAI:", response)
    print("-" * 50)