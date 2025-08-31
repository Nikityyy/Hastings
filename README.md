# Hastings Tokenizer

**Hastings** is a modern, optimized tokenizer for Large Language Models, developed as part of a suite of truly open-source AI tools.

The name is inspired by the historic town of Hastings, England, and is a playful nod to the concept of "tokens" in LLMs.

This tokenizer is used for pre-training and finetuning the **Lille-140M** model.

---

## ✨ Features

*   **Modern Base:** Derived from the robust `gpt2` tokenizer provided by `tiktoken`.
*   **Optimized Vocabulary:** A vocabulary size of **32,768** tokens, balancing expressiveness and efficiency.
*   **Custom Special Tokens:** Includes dedicated special tokens essential for modern instruction-tuning and chat formats.
*   **Efficient & Fast:** Built on `tiktoken`'s fast Rust core for high-performance tokenization.
*   **Simple to Use:** Packaged as a single `Hastings.pkl` file for easy integration.

### Special Tokens

Hastings includes the following special tokens mapped to the upper end of the vocabulary range:

- **`<|pad|>`** (`32763`): Padding token for batching sequences.
- **`<|endoftext|>`** (`32764`): Standard end-of-text marker.
- **`<|assistant|>`** (`32765`): Marks the beginning of an AI response.
- **`<|user|>`** (`32766`): Marks the beginning of a user prompt.
- **`<|startoftext|>`** (`32767`): Marks the start of a conversation.

## 🚀 Usage

Using the Hastings tokenizer is straightforward. First, ensure you have `tiktoken` installed:

```bash
pip install tiktoken
```

Then, you can load and use the tokenizer in your Python code.

```python
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
```

## 🎨 How It Was Created

The Hastings tokenizer was created to provide a clean, modern vocabulary with essential control tokens for chat and instruction-following models. The process was as follows:

1.  **Base Vocabulary:** Started with the public `gpt2` vocabulary from `tiktoken`.
2.  **Vocabulary Truncation:** The vocabulary was intentionally truncated to 32,768 tokens. This smaller size significantly reduces the memory footprint of the model's layers, leading to faster training iterations and making it more accessible for local, resource-constrained environments.
3.  **Special Token Injection:** Five custom special tokens (`<|pad|>`, `<|startoftext|>`, etc.) were added at the end of the vocabulary.

The exact implementation can be found in the `create_tokenizer.py` script.

## 🛠️ The truly open-source repos

Hastings is a key component of my initiative to build and release a complete, truly open-source stack for language modeling. All components are designed to work together seamlessly.

*   **Tokenizer:** **[Hastings](https://github.com/Nikityyy/Hastings)** (this repository) - A modern, efficient tokenizer with a 32k vocabulary.
*   **Dataset:** **[Kyoto-Corpus](https://github.com/Nikityyy/Kyoto-Corpus)** - A high-quality, small-scale dataset for instruction tuning.
*   **Model:** **[lille](https://github.com/Nikityyy/lille)** - A powerful 130-million-parameter model trained from scratch using the Hastings tokenizer.
*   **Optimizer:** **[Sophia-Triton](https://github.com/Nikityyy/Sophia-Triton)** - A memory-efficient, Triton-based implementation of the SophiaG optimizer.
*   **Evaluations:** **[simple-eval](https://github.com/Nikityyy/simple-eval)** - A straightforward framework for evaluating model performance using an LLM as a Judge.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](.../LICENSE) file for details.
