import torch
import torch.nn.functional as F

class RPCA_model(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim, dropout, batchnorm):
        super(RPCA_model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.nn = torch.nn.BatchNorm1d(num_features)
        self.convs.append(torch.nn.Linear(num_features, hidden_dim))
        # self.convs.append(torch.nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(torch.nn.Linear(hidden_dim, num_classes))  #
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters()) + list(self.convs[-1:].parameters())
        self.dropout = dropout
        self.batchnorm = batchnorm

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.batchnorm == True:
            x = self.nn(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.convs[0](x))
        for i, con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(con(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x)
        return F.log_softmax(x, dim=1)

