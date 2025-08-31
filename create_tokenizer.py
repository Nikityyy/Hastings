import tiktoken_ext.openai_public
import pickle

# --- 1. Load the Base Tokenizer ---
# We start with the standard GPT-2 tokenizer configuration. This gives us a robust
# and well-tested foundation for mergeable ranks and special token handling.
hastings = tiktoken_ext.openai_public.gpt2()

# --- 2. Define Vocabulary Size and Special Tokens ---
# The vocabulary size is set to 32,768 (2^15). This is an efficient size for
# modern GPUs and significantly reduces the model's embedding layer size,
# leading to faster training and a smaller memory footprint.
new_vocab_size = 32768

# Define the new special tokens and assign them unique integer IDs starting
# from the end of the vocabulary range. This is a common convention that keeps
# them separate from regular tokens.
new_special_tokens = {
    "<|pad|>": new_vocab_size - 5,
    "<|endoftext|>": new_vocab_size - 4,
    "<|assistant|>": new_vocab_size - 3,
    "<|user|>": new_vocab_size - 2,
    "<|startoftext|>": new_vocab_size - 1
}

# --- 3. Filter and Truncate the Vocabulary ---
# Get the original mergeable ranks from the gpt2 tokenizer.
mergeable_ranks = hastings['mergeable_ranks']

# The original gpt2 tokenizer includes "<|endoftext|>". To avoid conflicts,
# we filter it out from the base vocabulary before re-ranking.
filtered_mergeable_ranks = {k: v for k, v in mergeable_ranks.items() if k not in new_special_tokens}

# Create a new dictionary for the truncated and re-ranked vocabulary.
new_mergeable_ranks = {}

# The rank limit is the total vocab size minus the number of special tokens we're adding.
# This defines the space available for regular word/subword tokens.
rank_limit = new_vocab_size - len(new_special_tokens)

# Iterate through the filtered gpt2 ranks and assign new, sequential ranks (from 0 upwards).
# We stop once we hit our rank_limit, effectively truncating the vocabulary to the most
# common/important tokens from the original set.
for i, (token, rank) in enumerate(filtered_mergeable_ranks.items()):
    if i >= rank_limit:
        break
    new_mergeable_ranks[token] = i

# --- 4. Update and Save the Final Tokenizer ---
# Update the tokenizer dictionary with our new custom configuration.
hastings['special_tokens'] = new_special_tokens
hastings['explicit_n_vocab'] = new_vocab_size
hastings['name'] = 'Hastings'
hastings['mergeable_ranks'] = new_mergeable_ranks

# Serialize the final tokenizer dictionary to a .pkl file for easy loading.
with open('Hastings.pkl', 'wb') as f:
    pickle.dump(hastings, f)

print("Hastings tokenizer created and saved successfully to Hastings.pkl.")
