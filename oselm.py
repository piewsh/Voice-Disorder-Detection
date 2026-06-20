mport torch
import torch.nn as nn

class OSELM(nn.Module):
    def __init__(self, input_len, hidden_num, output_len, device):
        """
        Initialize the OSELM class with the number of input nodes, hidden nodes, and output nodes.

        Args:
        - input_len (int): Number of input nodes.
        - hidden_num (int): Number of hidden nodes.
        - output_len (int): Number of output nodes.
        - device (torch.device): Device to run the computations (CPU or GPU).
        """
        super(OSELM, self).__init__()
        self.input_len = input_len
        self.hidden_num = hidden_num
        self.output_len = output_len
        self.device = device

        # Initialize weights and biases
        self.W = nn.Parameter(torch.randn(input_len, hidden_num), requires_grad=False).to(self.device)
        self.b = nn.Parameter(torch.randn(hidden_num), requires_grad=False).to(self.device)
        self.beta = nn.Parameter(torch.zeros(hidden_num, output_len), requires_grad=False).to(self.device)
        self.P = nn.Parameter(torch.eye(hidden_num), requires_grad=False).to(self.device)
        self.init_flag = False

    def forward(self, x):
        """
        Forward pass for the OSELM model.

        Args:
        - x (torch.Tensor): Input tensor.

        Returns:
        - H (torch.Tensor): Transformed data after applying the activation function.
        """
        H = torch.sigmoid(torch.matmul(x, self.W) + self.b)
        return H

    def train_model(self, x, t):
        """
        Train the OSELM model with input data and target values.

        Args:
        - x (torch.Tensor): Input data.
        - t (torch.Tensor): Target values (labels).
        """
        H = self.forward(x)
        t = t.float()

        if not self.init_flag:
            t_P0 = torch.inverse(torch.matmul(H.T, H) + torch.eye(self.hidden_num, device=self.device) * 0.01)
            self.P.data = t_P0
            self.beta.data = torch.matmul(torch.matmul(self.P, H.T), t)
            self.init_flag = True
        else:
            eye = torch.eye(H.size(0), device=self.device)
            t_P1 = self.P - torch.matmul(
                torch.matmul(
                    torch.matmul(torch.matmul(self.P, H.T), torch.inverse(eye + torch.matmul(torch.matmul(H, self.P), H.T))),
                    H
                ), self.P
            )
            self.P.data = t_P1
            t_beta1 = self.beta + torch.matmul(torch.matmul(t_P1, H.T), (t - torch.matmul(H, self.beta)))
            self.beta.data = t_beta1

    def test_model(self, x, t=None):
        """
        Test the OSELM model with input data and optional target values.

        Args:
        - x (torch.Tensor): Input data.
        - t (torch.Tensor, optional): Target values for evaluating accuracy.

        Returns:
        - If `t` is provided, returns accuracy and predictions.
        - Otherwise, returns only predictions.
        """
        if not self.init_flag:
            raise Exception("Model not trained. Please train the model before testing.")

        H_test = self.forward(x)
        fx = torch.matmul(H_test, self.beta)

        if t is not None:
            accuracy = (torch.argmax(fx, dim=1) == torch.argmax(t, dim=1)).float().mean().item()
            return accuracy, fx
        else:
            return fx
