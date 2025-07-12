#include <torch/extension.h>
#include <vector>

// 必须加 extern "C"，与 .cu 文件一致，防止符号名 mismatch
extern "C" {
    void resample2d_kernel_forward(torch::Tensor input1, torch::Tensor input2, torch::Tensor output, int kernel_size, bool bilinear);
    void resample2d_kernel_backward(torch::Tensor input1, torch::Tensor input2, torch::Tensor gradOutput,
                                    torch::Tensor gradInput1, torch::Tensor gradInput2, int kernel_size, bool bilinear);
}

void forward(torch::Tensor input1, torch::Tensor input2, torch::Tensor output, int kernel_size, bool bilinear) {
    resample2d_kernel_forward(input1, input2, output, kernel_size, bilinear);
}

void backward(torch::Tensor input1, torch::Tensor input2, torch::Tensor gradOutput,
              torch::Tensor gradInput1, torch::Tensor gradInput2, int kernel_size, bool bilinear) {
    resample2d_kernel_backward(input1, input2, gradOutput, gradInput1, gradInput2, kernel_size, bilinear);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Resample2d forward");
    m.def("backward", &backward, "Resample2d backward");
}
