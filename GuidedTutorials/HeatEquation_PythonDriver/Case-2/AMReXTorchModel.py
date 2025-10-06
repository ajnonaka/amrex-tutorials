class AMReXTorchModel(AMReXModelBase, nn.Module):
    """Full PyTorch nn.Module compatibility with Function parameter handling"""

    def __init__(self, **kwargs):
        AMReXModelBase.__init__(self, **kwargs)
        nn.Module.__init__(self)

        # Register any learnable parameters if needed
        self.register_parameter('scale', nn.Parameter(torch.ones(1)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(1)))

    def forward(self, x):
        """Enhanced forward with learnable parameters"""
        # Get base simulation results
        outputs = AMReXModelBase.forward(self, x)

        # Apply learnable transformations if using PyTorch
        if isinstance(outputs, torch.Tensor):
            outputs = outputs * self.scale + self.bias

        return outputs

    def train_step(self, inputs, targets, optimizer, loss_fn):
        """Example training step for PyTorch workflows"""
        optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()


