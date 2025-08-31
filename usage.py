import tiktoken
import pickle

# 1. Load the tokenizer data from the pickle file
try:
    with open('Hastings.pkl', 'rb') as f:
        tokenizer_data = pickle.load(f)
except FileNotFoundError:
    print("Error: 'Hastings.pkl' not found. Make sure the file is in the correct path.")
    exit()

# 2. Initialize the tiktoken Encoding object
# The 'name' key is popped as it's used for initialization, not as a parameter.
enc = tiktoken.core.Encoding(
    name=tokenizer_data.pop('name'),
    **tokenizer_data
)

print(f"Tokenizer '{enc.name}' loaded successfully.")
print(f"Vocabulary size: {enc.n_vocab}")
print("-" * 30)

# --- Example Usage ---
original_text = "<|startoftext|><|user|>Hello, world!<|assistant|>Hey there!<|endoftext|>"

# 3. Encode text, allowing all special tokens
encoded_tokens = enc.encode(original_text, allowed_special="all")

# 4. Decode tokens back to text
decoded_text = enc.decode(encoded_tokens)

print(f"Original Text:  {original_text}")
print(f"Encoded Tokens: {encoded_tokens}")
print(f"Decoded Text:   {decoded_text}")
