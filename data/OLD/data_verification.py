import json

with open("data_test.json") as f:
    data = json.load(f)

for idx, problem in enumerate(data):
    unique_types = set()
    for job in problem["Jobs"]:
        for op in job:
            for time, machine in op:
                unique_types.add((time, machine))
    print(f"Problem {idx}: {len(unique_types)} unikalnych typów operacji")