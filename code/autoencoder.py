import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(embedding_size, 1)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        input_squeezed = self.linear(input)
        output, hidden = self.gru(input_squeezed.view(input_squeezed.size()[:-1]), hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, steps):
        output = []
        for i in range(steps):
            hidden = self.gru(input, hidden)
            input = self.out(hidden)
            output.append(input)
        return torch.stack(output, 0)

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size, device=device)


def train_autoencoder(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
                      max_length):
    encoder_hidden = encoder.initHidden(input_tensor.shape[1])

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    encoder_output, encoder_hidden = encoder(
        input_tensor, encoder_hidden)
    encoder_outputs = encoder_output[0, 0]

    decoder_input = torch.zeros(target_tensor.size()[1:], device=device, dtype=torch.float)
    # decoder_input = torch.cat([decoder_input, target_tensor[:-1]], 0)
    decoder_hidden = encoder_hidden[0]

    decoder_output = decoder(
        decoder_input, decoder_hidden, steps=target_tensor.shape[0])

    loss = criterion(decoder_output, target_tensor)
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length
