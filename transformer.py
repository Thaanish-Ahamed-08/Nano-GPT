import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32
context_length = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384
num_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

#########################################################################

with open('./stories.txt', 'r', encoding='utf-8') as file:
    data = file.read()

chars = sorted(list(set(data)))
vocab_size = len(chars)

#mapping for each characters
stoi = {st:idx for idx, st in enumerate(chars)}
itos = {idx:st for idx, st in enumerate(chars)}
encode = lambda x: [stoi[char] for char in x]
decode = lambda x: ''.join([itos[idx] for idx in x])

#Train and Val data splits
tensor_data = torch.tensor(encode(data), dtype=torch.long)
n = int(0.9*len(tensor_data))
train_data = tensor_data[:n]
val_data = tensor_data[n:]

#DataLoader
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    xb = torch.stack([data[i:i+context_length] for i in ix])  #Stack a one dimensional context_length for batch_size vertically to get 4,8 tensor
    yb = torch.stack([data[i+1:i+context_length+1] for i in ix])

    return xb, yb

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  #setting model to evaluation mode
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

############################################################################
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length,context_length))) 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        wei = q @ k.transpose(-2,-1) * (k.shape[-1]**-0.5)  # Attention scores
        wei = wei.masked_fill(self.tril[:T,:T]==0, float('-inf'))  #[:T,:T] => want this to be T,T dimension, masked the future words
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v  #(B,T,T)*(B,T,head_size) => (B,T,head_size) matrix
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(
            Head(head_size) for _ in range(num_heads)
        )
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        stack = []
        for h in self.heads:
            stack.append(h(x))
        out = torch.cat(stack,dim=-1) #stacking last dimension because head_size is at last dim
        out = self.dropout(self.proj(out)) #just a linear transformation for residual connections
        return out
    
class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),  #residual connections linear transformation
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    '''communciation followed by computation
    communication: Attention and computation: Linear or Feed Forward'''
    
    def __init__(self, n_embed, num_heads):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, head_size)  #Multi head self attention
        self.ffwd = FeedForward(n_embed) #FeedForward
        self.ln1 = nn.LayerNorm(n_embed)  #layernorm
        self.ln2 = nn.LayerNorm(n_embed)  #layernorm

    def forward(self,x):
        x = x + self.sa(self.ln1(x))    #the reason for the x +  is residual connections for the gradients to flow, if its plus, the gradient will be distributed across the leaf nodes during backpropagation
        x = x + self.ffwd(self.ln2(x))   #same reason here, refer transformer architecture
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embed)
        self.position_embedding_table = nn.Embedding(context_length, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, num_heads=num_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)  #Layer norm different from batch norm, batch norm, applied across columns or features, while layer norm applied for a token or row
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B,T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  #the shpe: (B,T,n_embed) => torch.Size([4, 8, 32])
        position_embedding = self.position_embedding_table(torch.arange(T)) #(T,C) -> Matrix
        x = token_embeddings + position_embedding #(B,T,C) + (T,C) => (B,T,C)+(B,T,C)
        x = self.blocks(x)  #transformer blocks
        x = self.ln_f(x) #final layer norm
        logits = self.lm_head(x)   #(B,T,vocab_size) -> vector

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #Negative log likelihood loss => 
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_con = idx[:,-context_length:] #try to pass only the context length words as inputs, not beyond than that
            logits,loss = self(idx_con)  #idx shape(B,T)
            logits = logits[:,-1,:]  #last time dimension logits that is we need to predict
            probs = F.softmax(logits, dim=-1)  #softmax will be applied for every column, thats why dim = -1
            idx_next = torch.multinomial(probs, replacement=True, num_samples=1) #(B,1)
            idx = torch.cat((idx, idx_next), dim=-1) #(B,T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-3) #since it is small model high lr is fine

for iter in range(max_iters):

    #every once evaulate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb,yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
