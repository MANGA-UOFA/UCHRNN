import torch
import torch.nn as nn

class HRNNtagger(nn.Module):
	def __init__(self, embedding_dim, hidden_dim):
		super(HRNNtagger, self).__init__()
		self.hidden_dim = hidden_dim

		self.rnn11 = nn.RNNCell(embedding_dim, hidden_dim)
		self.rnn12 = nn.RNNCell(embedding_dim, hidden_dim)
		self.rnn21 = nn.RNNCell(hidden_dim, hidden_dim)
  
		self.hidden2tag = nn.Linear(hidden_dim+hidden_dim+embedding_dim, 1)
		self.sig = nn.Sigmoid()

		self.h_init = torch.nn.Parameter(torch.zeros(hidden_dim))
		self.h1_actual = torch.zeros(hidden_dim)
		self.h2_actual = torch.zeros(hidden_dim)
	
	# def init_hidden(self, batch_size):
	# 	# initialize the hidden state and the cell state to zeros
	# 	return torch.zeros(batch_size, self.hidden_dim)
		
	def forward(self, bert_sent):
		batch_size, max_seq_len, hidden_dim = bert_sent.shape
		assert hidden_dim == self.hidden_dim
		output_seq = torch.zeros((batch_size, max_seq_len, 1)).type_as(bert_sent)
		tag_rep_seq = torch.zeros((batch_size, max_seq_len, 1)).type_as(bert_sent)
		h2_matrix = torch.zeros((batch_size, max_seq_len, self.hidden_dim)).type_as(bert_sent)

		h_init = self.h_init.expand(batch_size, -1).type_as(bert_sent)
		h1_actual = self.h1_actual.expand(batch_size, -1).type_as(bert_sent)
		h2_actual = self.h2_actual.expand(batch_size, -1).type_as(bert_sent)

		for t in range(max_seq_len):
			entry = bert_sent[:,t,:]

			tag_rep = self.hidden2tag(torch.cat((h1_actual, h2_actual, entry), dim=1))	
			output = self.sig(tag_rep)

			h11 = self.rnn11(entry, h1_actual)      		# low nocut
			h12 = self.rnn12(entry, h_init)					# low cut
			
			h22 = h2_actual									# high nocut
			h21 = self.rnn21(h1_actual, h2_actual)			# high cut

			h1_actual = torch.mul(h11, 1-output) + torch.mul(h12, output)
			h2_actual = torch.mul(h22, 1-output) + torch.mul(h21, output)

			output_seq[:,t,:] = output
			h2_matrix[:,t,:] = h2_actual
			tag_rep_seq[:,t,:] = tag_rep

		output_seq = torch.squeeze(output_seq, dim=-1)
		return output_seq, h2_matrix, tag_rep_seq

# class HRNNCutoffProjection(nn.Module):
# 	def __init__(self, max_len):
# 		super(HRNNCutoffProjection, self).__init__()
# 		self.linear = nn.Linear(max_len, max_len, bias=True)
# 		self.sig = nn.Sigmoid()
	
# 	def forward(self, input):
# 		return self.sig(self.linear(input))

# class ProjectionForAttention(nn.Module):
# 	def __init__(self, max_len, num=6):
# 		super(ProjectionForAttention, self).__init__()
# 		self.models = nn.ModuleList([HRNNCutoffProjection(max_len) for i in range(num)])
	
# 	def forward(self, input, index):
# 		return self.models[index](input)


if __name__ == "__main__":
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(device)

	batch_size, max_length = 16, 64
	embedding_dim, hidden_dim = 768,768
	hrnn = HRNNtagger(embedding_dim, hidden_dim).to(device)

	test_input = torch.randn(batch_size, max_length, embedding_dim).to(device)
	output_seq, h2_matrix = hrnn(test_input)

	proj = ProjectionForAttention(max_length, 6).to(device)

	print("upper level h2_matrix:", h2_matrix)
	print("m:", output_seq)
	
	out = torch.zeros(6, 16, 64).to(device)
	for i in range(6):
		out[i, :, :] = proj(output_seq, i)
	print(out)


