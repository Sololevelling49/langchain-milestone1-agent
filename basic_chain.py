from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load GPT-2
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=pipe)

# Prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple words:"
)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run
response = chain.run("LangChain")

print("\nBasic Chain Response:\n")
print(response)