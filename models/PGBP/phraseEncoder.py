import torch
import torch.nn as nn

class PhraseEncodeNet(nn.Module):

    def __init__(self, dim):
        super(PhraseEncodeNet, self).__init__()
        self.unigram_conv = nn.Conv1d(dim, dim, 1, stride=1, padding=0)
        self.bigram_conv  = nn.Conv1d(dim, dim, 2, stride=1, padding=1, dilation=2)
        self.trigram_conv = nn.Conv1d(dim, dim, 3, stride=1, padding=2, dilation=2)
        self.txt_linear = nn.Linear(dim * 3, dim)
        # padding,dilation设定保证L不变
    def forward(self, x):
        bs, _, dimc = x.size()
        words = x.transpose(-1, -2)  # B, C, L
        unigrams = self.unigram_conv(words)
        bigrams  = self.bigram_conv(words)  # B, C, L
        trigrams = self.trigram_conv(words)
        phrase = torch.cat((unigrams, bigrams, trigrams), dim=1)
        phrase = phrase.transpose(-1, -2).view(bs, -1, dimc * 3)
        phrase = self.txt_linear(phrase)
        return phrase
    
    