from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline

# =========================
# 1️⃣ Load Small Instruction Model
# =========================

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    do_sample=False,
)

llm = HuggingFacePipeline(pipeline=pipe)

# =========================
# 2️⃣ Define Tool Manually
# =========================

def multiply_numbers(input_text: str):
    try:
        a, b = input_text.split("*")
        return str(int(a.strip()) * int(b.strip()))
    except:
        return "Invalid format. Use: 45 * 8"

# =========================
# 3️⃣ Simple Tool Logic (NO Agent)
# =========================

question = "What is 45 * 8?"

# Let model decide if multiplication exists
if "*" in question:
    expression = question.replace("What is", "").replace("?", "").strip()
    result = multiply_numbers(expression)
    final_answer = f"The answer is {result}"
else:
    final_answer = llm.invoke(question)

# =========================
# 4️⃣ Output
# =========================

print("\n✅ Final Answer:\n")
print(final_answer)