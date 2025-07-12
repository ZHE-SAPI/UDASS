import torch
import resample2d_package.resample2d_cuda as resample2d_cuda


class Resample2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, kernel_size=1, bilinear=True):
        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        ctx.bilinear = bilinear
        output = torch.zeros_like(input1)
        resample2d_cuda.forward(input1, input2, output, kernel_size, bilinear)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1 = torch.zeros_like(input1)
        grad_input2 = torch.zeros_like(input2)
        resample2d_cuda.backward(
            input1, input2, grad_output,
            grad_input1, grad_input2,
            ctx.kernel_size, ctx.bilinear
        )
        return grad_input1, grad_input2, None, None


def resample2d(input1, input2, kernel_size=1, bilinear=True):
    """
    Functional interface for resample2d operation.
    """
    return Resample2dFunction.apply(input1, input2, kernel_size, bilinear)


class Resample2d(torch.nn.Module):
    """
    Module wrapper for Resample2dFunction.
    """
    def __init__(self, kernel_size=1, bilinear=True):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.bilinear = bilinear

    def forward(self, input1, input2):
        return Resample2dFunction.apply(input1, input2, self.kernel_size, self.bilinear)
