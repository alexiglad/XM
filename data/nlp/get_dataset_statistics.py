import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from bigbench_dataloader import BigBenchDataset
from ai2arc_dataloader import AI2ArcDataset
from gsm8k_dataloader import GSM8KDataset
from squad_dataloader import SQuADDataset
from planbench_dataloader import PlanBenchDataset

backbone = "EleutherAI/gpt-neox-20b"
tokenizer = AutoTokenizer.from_pretrained(backbone)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="")
parser.add_argument("--execution_mode", type=str, default="finetune")
args = parser.parse_args()

datasets_q_mean = []
datasets_q_std = []
datasets_q_max = []
datasets_q_min = []
datasets_a_mean = []
datasets_a_std = []
datasets_a_max = []
datasets_a_min = []
dataset_names = []
data_counts = []

for dataset_name in [BigBenchDataset, AI2ArcDataset, GSM8KDataset, SQuADDataset]:
    try:
        dataset = dataset_name(args, "train", "all")
    except:
        dataset = dataset_name(args, "train")

    token_q = [len(tokenizer.tokenize(data.split("[[Answer]]:")[0].split("[[Question]]:")[1])) for data in dataset]
    token_a = [len(tokenizer.tokenize(data.split("[[Answer]]:")[1])) for data in dataset]

    datasets_q_mean.append(np.mean(token_q))
    datasets_q_std.append(np.std(token_q))
    datasets_q_max.append(np.max(token_q))
    datasets_q_min.append(np.min(token_q))

    datasets_a_mean.append(np.mean(token_a))
    datasets_a_std.append(np.std(token_a))
    datasets_a_max.append(np.max(token_a))
    datasets_a_min.append(np.min(token_a))

    data_counts.append(len(dataset))
    dataset_names.append(dataset_name.__name__)

x = np.arange(len(dataset_names))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.bar(x - width / 2, datasets_q_mean, width, label='Q Mean', color='blue')
bar2 = ax.bar(x + width / 2, datasets_a_mean, width, label='A Mean', color='orange')

def add_std_bars(bars, stds):
    for bar, std in zip(bars, stds):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'±{round(std, 2)}', ha='center', va='bottom')

add_std_bars(bar1, datasets_q_std)
add_std_bars(bar2, datasets_a_std)

ax.set_xlabel('Dataset')
ax.set_ylabel('Token Count')
ax.set_title('Token Count Statistics for Questions and Answers across Datasets')
ax.set_xticks(x)
ax.set_xticklabels(dataset_names, rotation=45, ha="right")
ax.legend()

plt.tight_layout()
plt.savefig('token_count_statistics.png', dpi=300)

data = {
    'Dataset': dataset_names,
    'Q Mean': datasets_q_mean,
    'Q Std': datasets_q_std,
    'Q Max': datasets_q_max,
    'Q Min': datasets_q_min,
    'A Mean': datasets_a_mean,
    'A Std': datasets_a_std,
    'A Max': datasets_a_max,
    'A Min': datasets_a_min,
    'Data Count': data_counts
}

df = pd.DataFrame(data)
df.to_csv('token_count_statistics.csv', index=False)
print("CSV file 'token_count_statistics.csv' has been created.")
