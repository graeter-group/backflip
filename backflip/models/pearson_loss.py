# Copyright (c) 2025 Max Planck Institute for Polymer Research
# Licensed under the MIT license.

import torch
import torch.nn as nn

class PearsonCorrelationLoss(nn.Module):
    def __init__(self, dim=-1):
        """
        Pearson Correlation Coefficient Loss
        Returns the negative Pearson correlation coefficient between input and target tensors.
        Args:
            dim (int): Dimension along which to compute the correlation.
        Forward method accepts tensors of shape (*batch_dims, N, target_dims*), where N is at dim, and computes the correlation along the N dimension, i.e. returns a tensor of shape (*batch_dims, target_dims*).
        """
        super(PearsonCorrelationLoss, self).__init__()
        self.dim = dim

        self.eps = 1e-3

    def forward(self, x, y):
        # Ensure the inputs have the same shape
        assert x.shape == y.shape, "Input tensors must have the same size"

        # add some slight noise to avoid errors because of zero std
        x = x + torch.randn_like(x) * self.eps
        y = y + torch.randn_like(y) * self.eps
        
        # Mean centering the variables
        x_mean = torch.mean(x, dim=self.dim, keepdim=True)
        y_mean = torch.mean(y, dim=self.dim, keepdim=True)
        x_cent = x - x_mean
        y_cent = y - y_mean
        
        # Calculating the covariance between x and y
        cov_xy = torch.mean(x_cent * y_cent, dim=self.dim)
        
        # Calculating the standard deviations of x and y
        std_x = torch.sqrt(torch.mean(x_cent ** 2, dim=self.dim))
        std_y = torch.sqrt(torch.mean(y_cent ** 2, dim=self.dim))
        
        # Calculating the Pearson Correlation Coefficient
        pearson_r = cov_xy / (std_x * std_y + self.eps)
        
        # Loss is negative Pearson correlation coefficient
        return -pearson_r