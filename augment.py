import random

def swap(val, vocab_size):
    return random.randint(3, vocab_size) # 0,1,2 are special tokens

def insert(val, vocab_size):
    return [random.randint(3, vocab_size), val] # 0,1,2 are special tokens

def delete(val):
    return []

def augment(val, tokenizer, prob):
    if val in [0,1,2]: return val
    if random.random() < prob:
        aug_func = random.choice([0, 1, 2])
        if aug_func == 0:
            val = swap(val, vocab_size=tokenizer.vocab_size)
        if aug_func == 1:
            val = insert(val, vocab_size=tokenizer.vocab_size)
        if aug_func == 2:
            val = delete(val)
    return val

def flatten(t):
    flat_list = []
    for x in t:
        if type(x) == list:
            for item in x:
                flat_list.append(item)
        else: flat_list.append(x)
    return flat_list

def augment_function(ex, tokenizer, prob=0.1):
    example = ex.copy()
    ids = example['input_ids']
    example['input_ids'] = flatten([augment(x, tokenizer, prob) for x in ids])
    attn = len([x for x in example['input_ids'] if x != 1])
    other = len(example['input_ids']) - attn
    example['attention_mask'] = [1] * attn + [0] * other
    return example