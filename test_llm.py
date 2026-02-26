from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline

model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=False   # deterministic
)

llm = HuggingFacePipeline(pipeline=pipe)

response = llm.invoke(
    "Explain what LangChain is in simple words:"
)

print("\nModel Response:\n")
print(response)