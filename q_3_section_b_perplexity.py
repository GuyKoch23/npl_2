import string
import random
import re
import requests
import unidecode

url = "https://github.com/tau-nlp-course/NLP_HW2/raw/main/data/shakespeare.txt"


all_characters = string.printable
n_characters = len(all_characters)  # our vocabulary size (|V| from the handout)

dataset_as_string = unidecode.unidecode(requests.get(url).content.decode())
n_chars_in_dataset = len(dataset_as_string)
print(f"Total number of characters in our dataset: {n_chars_in_dataset}")

chunk_len = 400


def random_chunk():
    start_index = random.randint(0, n_chars_in_dataset - chunk_len)
    end_index = start_index + chunk_len + 1
    return dataset_as_string[start_index:end_index]


print(random_chunk())

import torch
import torch.nn as nn
from torch.autograd import Variable
import string
import random


class OurModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(OurModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            input_size, hidden_size
        )  # In the terms of the handout, here d = D_h
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, input_, hidden):
        # General instructions:
        # Pass the embedded input through the GRU and use the output layer to get the next character distribution.
        # return that distribution and the next hidden state.
        # You may need to play around with the dimensions a bit until you get it right. Dimension-induced frustration is good for you!
        # -------------------------
        embedded = self.embedding(input_)
        embedded = embedded.view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.output_layer(output.view(1, -1))
        # -------------------------
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.num_layers, 1, self.hidden_size))


"""### Creating the Training Examples

Each chunk will be turned into a tensor by looping through the characters of the string and looking up the index of each character in `all_characters`.
"""


# Turn a string into list of longs
def chars_to_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)


print(chars_to_tensor("abcDEF"))

"""Now we can assemble a pair of input and target tensors (i.e. a single training example) for training, from a random chunk. The input will be all characters *except the last*, and the target will be all characters *except the first*. So if our chunk is "abc" the input will correspond to "ab" while the target is "bc"."""


def random_training_set():
    chunk = random_chunk()
    inp = chars_to_tensor(chunk[:-1])
    target = chars_to_tensor(chunk[1:])
    return inp, target


"""### Evaluating

To evaluate the network we will feed one character at a time, use the outputs of the network as a probability distribution for the next character, and repeat. To start generation we pass a priming string to start building up the hidden state, from which we then generate one character at a time.
"""

import torch.nn.functional as F


def evaluate(prime_str="A", predict_len=100, temperature=0.8):
    hidden = model.init_hidden()
    prime_input = chars_to_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = model(prime_input[p], hidden)
    inp = prime_input[-1]

    for p in range(predict_len):
        output, hidden = model(inp, hidden)

        # Sample from the network as a multinomial distribution
        output_dist = F.softmax(output / temperature, dim=-1)
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = chars_to_tensor(predicted_char)

    return predicted


"""### Training

The main training function
"""


def train(inp, target):
    hidden = model.init_hidden()
    model.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = model(inp[c], hidden)
        loss += criterion(output, target[c].view(-1))

    loss.backward()
    optimizer.step()

    return loss.item() / chunk_len


"""A helper to print the amount of time passed:"""

import time, math


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return f"{m}m {math.floor(s)}s"


# DO NOT DELETE THIS CELL

"""Define the training parameters, instantiate the model, and start training:"""

n_epochs = 2000
print_every = 100
plot_every = 10
hidden_size = 100  # (D_h from the handout)
num_layers = 1
lr = 0.005

model = OurModel(n_characters, hidden_size, n_characters, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set())
    loss_avg += loss

    if epoch % print_every == 0:
        print(
            f"[time elapsed: {time_since(start)}  ;  epochs: {epoch} ({epoch / n_epochs * 100}%)  ;  loss: {loss:.4}]"
        )
        print(evaluate("Wh", 200), "\n")  # generate text starting with 'Wh'

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0

# OUR CODE------------------------------------------------------

import numpy as np
import unidecode


def perplexity(text_file):
    with open(text_file, "r", encoding="utf-8") as file:
        text = unidecode.unidecode(file.read())
    temperature = 0.8
    total_log_probability = 0
    hidden = model.init_hidden()
    for i in range(len(text) - 1):
        try:
            current_char_tensor = chars_to_tensor(text[i])
            output, hidden = model(current_char_tensor, hidden)
            output = output / temperature
            probabilities = torch.softmax(output, dim=-1)

            next_char_index = all_characters.index(text[i + 1])
            next_char_probability = probabilities[0][next_char_index].item()

            total_log_probability += np.log2(next_char_probability)

        except Exception as e:
            print(f"Error processing character at position {i}: {e}")
            continue
    avg_log_probability = total_log_probability / len(text)
    perplexity = 2 ** (-1 * avg_log_probability)
    return perplexity


print(
    "Perplexity of shakespeare_for_perplexity.txt: "
    + str(perplexity("shakespeare_for_perplexity.txt"))
)
print(
    "Perplexity of wikipedia_for_perplexity.txt: "
    + str(perplexity("wikipedia_for_perplexity.txt"))
)
