from transformers import pipeline

model = pipeline(task="text-generation", model="deepseek-ai/DeepSeek-R1")
response = model("How many trophies has the Portuguese national team won and when?")
print(response)
