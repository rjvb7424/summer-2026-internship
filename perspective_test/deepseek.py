from transformers import pipeline

model = pipeline(task="text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")
response = model("How many trophies has the Portuguese national team won and when?")
print(response)
