import re
model_path= "/home/llmserver/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B/snapshots/91d2aff3f957f99e4c74c962f2f408dcc88a18d8"
pattern = r"models--(.*?)--(.*?)/"

m = re.search(pattern, model_path)


print(f" >>> group0: {m.group(0)}")
print(f" >>> group1: {m.group(1)}")
