#!/usr/bin/env python
# -*- coding: utf-8 -*-



''' Inherent libs '''
import os


''' Third libs '''
import math
import numpy as np
import torch
import torch.nn as nn


''' Local libs '''



class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """[Encoding position for 1D space - range(0, length_of_position, 1)

        Args:
            channels ([int-even]): [The last dimension of the tensor you want to apply pos emb to.]
        """   

        super(PositionalEncoding1D, self).__init__()
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)
        

    def forward(self, positions_tensor):
        """[summary]

        Args:
            positions_tensor ([torch.tensor]): [A 3d tensor of size (batch_size, x, ch),
                                                x is the length of positions, thus this function will
                                                encode each position (range(0, x, 1)) of x]

        Raises:
            RuntimeError: [description]

        Returns:
            [torch.tensor]: [Positional Encoding Matrix of size (batch_size, x, ch)]
        """
        if len(positions_tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        _, x, orig_ch = positions_tensor.shape
        pos_x = torch.arange(x, device=positions_tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x,self.channels),device=positions_tensor.device).type(positions_tensor.type())
        emb[:,:self.channels] = emb_x

        return emb[None,:,:orig_ch]


class PositionalEncodingPermute1D(nn.Module):
    #  (batchsize, ch, x) instead of (batchsize, x, ch)

    def __init__(self, channels):

        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0,2,1)
        enc = self.penc(tensor)
        return enc.permute(0,2,1)

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        # channels is the desired dimension of the output tensor
        super(PositionalEncoding2D, self).__init__()
        channels = int(np.ceil(channels/2))
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, positions_tensor):
        """[Encoding position for 2D space - range(0, X_length_of_position, 1), range(0, Y_length_of_position, 1)]

        Args:
            positions_tensor ([torch.tensor]): [A 4d tensor of size (batch_size, x, y, ch),
                                                x is the width of positions and y is the height of positions, 
                                                thus this function will encode each position presented with (x_i, y_j),
                                                where x_i \in [0, x], y_j \in [0, y]],
                                                ch is the desired dimension used to store the encoded positions

        Raises:
            RuntimeError: [description]

        Returns:
            [torch.tensor]: [Positional Encoding Matrix of size (batch_size, x, y, ch),
                            the first 0:d/2 holds the position embedding for x_i while the d/2: holds the 
                            position encoding for y_j]
        

        Note: To use this function, we need to create an empty tensor first. Then, the ch for positions_tensor should 
                equals to the required channels so that it can hold the output. 
                p_enc_2d = PositionalEncoding2D(8)
                y = torch.zeros((1,6,2,8))
                print(p_enc_2d(y).shape) # (1, 6, 2, 8)
        """
        if len(positions_tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        _, x, y, orig_ch = positions_tensor.shape
        pos_x = torch.arange(x, device=positions_tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=positions_tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x,y,self.channels*2),device=positions_tensor.device).type(positions_tensor.type())
        emb[:,:,:self.channels] = emb_x
        emb[:,:,self.channels:2*self.channels] = emb_y

        return emb[None,:,:,:orig_ch]

class PositionalEncodingPermute2D(nn.Module):
    # Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)    
    def __init__(self, channels):
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0,2,3,1)
        enc = self.penc(tensor)
        return enc.permute(0,3,1,2)


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):

        super(PositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels/3))
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):

        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        _, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z

        return emb[None,:,:,:,:orig_ch]

class PositionalEncodingPermute3D(nn.Module):
    # Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)      
    def __init__(self, channels):

        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0,2,3,4,1)
        enc = self.penc(tensor)
        return enc.permute(0,4,1,2,3)
