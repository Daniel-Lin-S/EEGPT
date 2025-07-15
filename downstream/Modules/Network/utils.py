import torch.nn as nn
import torch
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class Conv2dWithConstraint(nn.Conv2d):
    """
    A 2D convolutional layer with weight constraint.
    It prevents the weights from exceeding a specified maximum L2 norm.

    Reference
    ----------
    Lawhern V J, Solon A J, Waytowich N R, et al.
    EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces[J].
    Journal of neural engineering, 2018, 15(5): 056013.
    """
    def __init__(
            self, *args,
            doWeightNorm = True,
            max_norm: float=1,
            **kwargs
        ):
        """
        Parameters
        ----------
        args : tuple
            Positional arguments for nn.Conv1d.
            e.g. (in_channels, out_channels, kernel_size).
        doWeightNorm : bool, optional
            If True, applies weight normalisation during the forward pass.
            Defaults to True.
            Otherwise, the model regresses to the original Conv1d layer.
        max_norm : float, optional
            The maximum L2 norm for the weights.
            Defaults to 1.
        kwargs : dict
            Additional keyword arguments for nn.Conv1d.
            e.g. stride, padding, dilation, groups, bias, padding_mode.
        """
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W)
            where B is batch size,
            and H, W are height and width of the input feature map.

        Returns
        -------
        torch.Tensor
            Output tensor. 
            With shape (B, out_channels, H_out, W_out)
            where H_out and W_out depend on the kernel size, stride, and padding.
        """
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class Conv1dWithConstraint(nn.Conv1d):
    """
    A 1D convolutional layer with weight constraint.
    It prevents the weights from exceeding a specified maximum L2 norm.

    Reference
    ----------
    Lawhern V J, Solon A J, Waytowich N R, et al.
    EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces[J].
    Journal of neural engineering, 2018, 15(5): 056013.
    """
    def __init__(
            self, *args,
            doWeightNorm: bool = True,
            max_norm: float=1,
            **kwargs
        ):
        """
        Parameters
        ----------
        args : tuple
            Positional arguments for nn.Conv1d.
            e.g. (in_channels, out_channels, kernel_size).
        doWeightNorm : bool, optional
            If True, applies weight normalisation during the forward pass.
            Defaults to True.
            Otherwise, the model regresses to the original Conv1d layer.
        max_norm : float, optional
            The maximum L2 norm for the weights.
            Defaults to 1.
        kwargs : dict
            Additional keyword arguments for nn.Conv1d.
            e.g. stride, padding, dilation, groups, bias, padding_mode.
        """
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, L)
            where B is batch size,
            and L is length of the sequence.

        Returns
        -------
        torch.Tensor
            Output tensor. 
            With shape (B, out_channels, L_out)
            where L_out depends on the kernel size, stride, and padding.
        """
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)

class LinearWithConstraint(nn.Linear):
    """
    A linear layer with weight constraint.
    Applies renormalisation to the weights during the forward pass to
    ensure that the L2 norm of the weights does not exceed a specified maximum norm.

    Attributes
    ----------
    max_norm : int
        The maximum L2 norm for the weights.
    doWeightNorm : bool
        If True, applies weight normalisation during the forward pass.
        if False, the model regresses to the original linear layer.
    """
    def __init__(
            self, *args,
            doWeightNorm: bool = True,
            max_norm: int=1, **kwargs):
        """
        Parameters
        ----------
        args : tuple
            Positional arguments for the parent class,
            which is nn.Linear.
            e.g. (512, 256) for a linear layer with
            512 input features and 256 output features.
        doWeightNorm : bool, optional
            If True, applies weight normalisation during the forward pass.
            Defaults to True.
        max_norm : int, optional
            The maximum L2 norm for the weights.
            Defaults to 1.
        """
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C)
            where B is batch size and C is number of features.
        """
        if self.doWeightNorm: 
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


def SMMDL_marginal(Cs,Ct):

    '''
    The SMMDL used in the CRGNet.
    Arg:
        Cs:The source input which shape is NxdXd.
        Ct:The target input which shape is Nxdxd.
    '''
    
    Cs = torch.mean(Cs,dim=0)
    Ct = torch.mean(Ct,dim=0)
    
    # loss = torch.mean((Cs-Ct)**2)
    loss = torch.mean(torch.mul((Cs-Ct), (Cs-Ct)))
    
    return loss

def SMMDL_conditional(Cs,s_label,Ct,t_label):
  
    '''
    The Conditional SMMDL of the source and target data.
    Arg:
        Cs:The source input which shape is NxdXd.
        s_label:The label of Cs data.
        Ct:The target input which shape is Nxdxd.
        t_label:The label of Ct data.
    '''
    s_label = s_label.reshape(-1)
    t_label = t_label.reshape(-1)
    
    class_unique = torch.unique(s_label)
    
    class_num = len(class_unique)
    all_loss = 0.0
    
    for c in class_unique:
        s_index = (s_label == c)
        t_index = (t_label == c)
        # print(t_index)
        if torch.sum(t_index)==0:
            class_num-=1
            continue
        c_Cs = Cs[s_index]
        c_Ct = Ct[t_index]
        m_Cs = torch.mean(c_Cs,dim = 0)
        m_Ct = torch.mean(c_Ct,dim = 0)
        loss = torch.mean((m_Cs-m_Ct)**2)
        all_loss +=loss
        
    if class_num == 0:
        return 0
    
    return all_loss/class_num   

