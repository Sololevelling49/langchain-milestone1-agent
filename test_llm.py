from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

pipe = pipeline(
    "text-generation",   # ✅ correct task
    model="google/flan-t5-base"
)
# Wrap in LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Test model
response = llm.invoke("Explain what LangChain is in simple words.")

print("\nModel Response:\n")
print(response)