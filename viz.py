import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("results.csv")
df["Model"] = df["Model"].str.replace("[a-z]", "")

fig = plt.figure(figsize=(50, 30))

steps = [2,4,6,8,10]
alphas = [0.01, 0.02, 0.04]

for i, alpha in enumerate(alphas):
    for j, step in enumerate(steps):
        sdf = df[(df["Steps"] == step) & (df["Alpha"] == alpha)]
        ax = fig.add_subplot(len(alphas), len(steps), i*len(steps)+j+1)
        ax.plot(sdf["Model"], sdf["ASR"], marker="o")
        ax.set_title(f"{step}-step, alpha={alpha}")


plt.savefig("test.png")


fig = plt.figure(figsize=(30, 20))
for i, alpha in enumerate(alphas):
    ax = fig.add_subplot(len(alphas)-1, 2, i+1)
    ax.set_title(f"Alpha={alpha}")
    for j, step in enumerate(steps):
        sdf = df[(df["Steps"] == step) & (df["Alpha"] == alpha)]
        ax.plot(sdf["Model"], sdf["ASR"], label=f"{step}-Step", marker="o")
    
    ax.legend()
    

plt.savefig("alpha.png")


fig = plt.figure(figsize=(20, 30))
for i, step in enumerate(steps):
    ax = fig.add_subplot(len(steps)-2, 2, i+1)
    ax.set_title(f"{step}-Step Attack")
    for j, alpha in enumerate(alphas):
        sdf = df[(df["Steps"] == step) & (df["Alpha"] == alpha)]
        ax.plot(sdf["Model"], sdf["ASR"], label=f"Alpha={alpha}", marker="o")
    
    ax.legend()
    

plt.savefig("step.png")


models = ["B", "BN", "D", "GN", "IN", "LN"]

fig = plt.figure(figsize=(20, 30))
for i, model in enumerate(models):
    ax = fig.add_subplot(3, 2, i+1)
    ax.set_title(f"Model: {model}")
    for j, step in enumerate(steps):
        sdf = df[(df["Steps"] == step) & (df["Model"] == model)]
        ax.plot(sdf["Alpha"], sdf["ASR"], label=f"step={step}", marker="o")
    ax.legend()

plt.savefig("model-step.png")

models = ["B", "BN", "D", "GN", "IN", "LN"]

fig = plt.figure(figsize=(20, 30))
for i, model in enumerate(models):
    ax = fig.add_subplot(3, 2, i+1)
    ax.set_title(f"Model: {model}")
    for j, alpha in enumerate(alphas):
        sdf = df[(df["Alpha"] == alpha) & (df["Model"] == model)]
        ax.plot(sdf["Steps"], sdf["ASR"], label=f"alpha={alpha}", marker="o")
    ax.legend()
plt.savefig("model-alpha.png")