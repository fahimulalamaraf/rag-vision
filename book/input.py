import torch
import torch.nn as nn

batch_size = 64
block_size = 8
learning_rate = 1e-5
max_iters = 1000000


with open('../weathuringheights.txt', 'r', encoding='utf-8') as file:
    text = file.read()



chars = sorted(list(set(text)))

v_size = len(chars)

string_to_int = {c: i for i, c in enumerate(chars)}
int_to_string = {i: c for i, c in enumerate(chars)}

encode = lambda s:[string_to_int[c] for c in s]

decode = lambda i:"".join(int_to_string[c.item()] for c in i)

data = torch.tensor(encode(text))


train_data = data[:int(len(data)*0.8)]
valid_data = data[int(len(data)*0.8):]

def get_batch(split):
    batch_data = train_data if split == 'train' else valid_data
    window = torch.randint(len(batch_data) - batch_size, (block_size,))
    x = torch.stack([batch_data[i:i + batch_size] for i in window])
    y = torch.stack([batch_data[i + 1:i + batch_size + 1] for i in window])
    return x, y


class TextGenModel(nn.Module):
    def __init__(self, v_size):
        super().__init__()
        self.token_embedding_Table = nn.Embedding(v_size, v_size)

    def forward(self, input, targets=None):
        scores = self.token_embedding_Table(input)

        if targets is None:
            loss = None

        else:
            B, T, C = scores.shape
            scores = scores.view(B*T, C)
            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(scores, targets)
        return scores, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            scores, loss = self.forward(index_cond)
            scores = scores[:, -1, :]
            probs = torch.nn.functional.softmax(scores, dim=-1)
            index_next = torch.multinomial(probs, num_samples=1)
            index = torch.cat((index, index_next), dim=-1)
        return index

m = TextGenModel(len(chars))
context_with_given_input = torch.tensor([encode('Heathcliff')], dtype=torch.long)
context = torch.zeros((1, 1), dtype=torch.long)

g_char = decode(torch.tensor(m.generate(context_with_given_input, max_new_tokens=100)[0].tolist()))
print('Generate_without_train: \n', '#'*100)
print(g_char)

optimizer = torch.optim.Adam(m.parameters(), lr=learning_rate)

for i in range(max_iters):
    xb, yb = get_batch('train')
    scores, mloss = m.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    mloss.backward()
    optimizer.step()
    if i % 1000 == 0:
        print(i, mloss.item())

print('\nafter training loss is:', mloss.item())

g_char = decode(torch.tensor(m.generate(context_with_given_input, max_new_tokens=10)[0].tolist()))
print('Generate after train: \n', '#'*100)
print(" ", g_char)