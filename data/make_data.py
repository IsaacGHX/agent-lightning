# Cell 1
import pandas as pd

train_file = "./data/train.parquet"
val_file = "./data/test.parquet"
mini_file = "./data/test_mini.parquet"
df = pd.read_parquet(train_file)
val_df = pd.read_parquet(val_file)
mini_df = pd.read_parquet(mini_file)

# Cell 2: This cell was not executed in the notebook, but is included here.
# Assuming the intention was to access the 10th row.
# In pandas, it's better to use .iloc for integer-location based indexing.
# print(df.iloc[9]) 

# Cell 3
print("Displaying the initial DataFrame 'df':")
print(df)

# Cell 4
df["extra_info"] = [
    {"idx": i, "ground_truth": gt}
    for i, gt in enumerate(df["result"])
]

val_df["extra_info"] = [
    {"idx": i, "ground_truth": gt}
    for i, gt in enumerate(val_df["result"])
]

mini_df["extra_info"] = [
    {"idx": i, "ground_truth": gt}
    for i, gt in enumerate(mini_df["result"])]

# Cell 5
print("\nDisplaying DataFrame 'df' after adding 'extra_info' column:")
print(df)

# Cell 6
df.to_parquet(train_file)
val_df.to_parquet(val_file)
mini_df.to_parquet(mini_file)

# Cell 7
print("\nDisplaying DataFrame 'df' after saving to parquet (content remains the same):")
print(df)

# Cell 8: This cell had no code in the original notebook.

# Cell 9: The output in the notebook indicates that the original 'extra_info' column was dropped at some point.
# To replicate the displayed output, we will drop it before printing.
# Note: The notebook shows a state where 'extra_info' is gone, but the code to drop it is not present.
# We infer this action to match the output.
if 'extra_info' in df.columns:
    df_no_extra_info = df.drop(columns=['extra_info'])
    print("\nDisplaying DataFrame 'df' as shown in the later notebook cell (without 'extra_info'):")
    print(df_no_extra_info)
else:
    print("\nDisplaying DataFrame 'df' as is:")
    print(df)


# Cell 10
from pathlib import Path

# Assuming rollout_data_dir is a file path
rollout_data_dir = "./rollout_data/rollout_data_1dot5B_0_test"
p = Path(rollout_data_dir)

print("\nDemonstrating pathlib Path operations:")
print(p)
print(p.parent)
print(p.parent.parent)

# Cell 11
Path(rollout_data_dir).mkdir(parents=True, exist_ok=True)
print(f"\nDirectory '{rollout_data_dir}' created or already exists.")


# Cell 12
from transformers import AutoTokenizer

# Select the tokenizer corresponding to the model, e.g., Qwen/Qwen2.5-7B-Instruct
model = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model)

text = '''Tickets to a certain concert sell for $20 each. The first 10 people to show up at the ticket booth received a 40% discount, and the next 20 received a 15% discount. If 48 people bought tickets to the concert, what was the total revenue from ticket sales?  Choose the most appropriate option:
A) $ 600
B) $ 820
C) $ 850
D) $ 980
E) $ 1,140 Output the answer when you are ready. The answer should be surrounded by three sharps (`###`), in the form of ### ANSWER: [SILENT] ###. 
'''

# Encode the text
tokens = tokenizer.encode(text, add_special_tokens=False)

# Number of tokens
print("\nTokenizing first text example:")
print("Token nums:", len(tokens))
# print("Tokens:", tokens) # Uncomment to see the full list of tokens

# Cell 13
text_2 = '''The query involves calculating the total revenue from ticket sales for a concert. The ticket price is $20, with different discount rates applied to different groups of people. Specifically, the first 10 people received a 40% discount, the next 20 people received a 15% discount, and the remaining people paid the full price.

### Necessary Skills and Tools

#### Skills
1. **Mathematical Calculation**: The ability to perform arithmetic operations such as multiplication, addition, and percentage calculations.
2. **Logical Reasoning**: The ability to break down the problem into smaller parts and solve it step by step.

#### Tools
1. **Generalist_Solution_Generator_Tool**: This tool can be used to perform the necessary calculations and logical reasoning to solve the problem.

### Explanation of Skills and Tools

#### Mathematical Calculation
- **Explanation**: The query requires calculating the revenue from different groups of ticket buyers. This involves multiplying the number of tickets sold by the respective discounted prices and then summing these amounts.
- **How it helps**: The skill of performing mathematical calculations is essential for determining the total revenue from each group of ticket buyers.

#### Logical Reasoning
- **Explanation**: The problem needs to be broken down into smaller parts to handle the different discount rates and the number of people in each group.
- **How it helps**: Logical reasoning helps in organizing the information and applying the correct discount rates to the appropriate number of people.

#### Generalist_Solution_Generator_Tool
- **Explanation**: This tool can be used to input the query and perform the necessary calculations and logical reasoning to determine the total revenue.
- **How it helps**: The tool simplifies the process by automating the calculations and ensuring accuracy. It can handle the step-by-step reasoning required to solve the problem.

### Additional Considerations
1. **Verification**: After using the tool, it is important to verify the calculations manually to ensure accuracy.
2. **Tool Limitations**: Be aware that the tool may provide hallucinated or incorrect responses, as mentioned in the metadata. Always cross-check the results.
3. **Complexity**: For more complex queries, breaking down the problem into smaller subtasks and using the tool multiple times may be necessary.

### ANSWER: [SILENT] ###

To solve the problem, we can use the Generalist_Solution_Generator_Tool to perform the necessary calculations. Here’s a step-by-step breakdown:

1. **Calculate the revenue from the first 1
'''

# Encode the text
tokens_2 = tokenizer.encode(text_2, add_special_tokens=False)

# Number of tokens
print("\nTokenizing second text example:")
print("Token nums:", len(tokens_2))
# print("Tokens:", tokens_2) # Uncomment to see the full list of tokens

# Cell 14: This cell had no code in the original notebook.