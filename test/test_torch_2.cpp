#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/version.h>

#include <iostream>
#include "fmtfallback.h"
// 将 SM 版本转换为核心数的辅助函数
int _ConvertSMVer2Cores(int major, int minor);

int main() {
    // 版本信息
    std::cout << "LibTorch 版本:" << TORCH_VERSION << std::endl;

    // 检查是否支持CUDA
    if (torch::cuda::is_available()) {
        std::cout << "CUDA 可用！使用 GPU" << std::endl;
        std::cout << "cuDNN 可用状态："
                  << (torch::cuda::cudnn_is_available() ? "true" : "false")
                  << std::endl;

        int device_count = torch::cuda::device_count();
        // cudaGetDeviceCount(&device_count);
        std::cout << "CUDA 设备数量:" << device_count << std::endl;

        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            std::cout << "设备 " << i << ": " << prop.name << std::endl;
            std::cout << "  总内存：" << prop.totalGlobalMem / (1024 * 1024)
                      << " MB" << std::endl;
            std::cout << "  多处理器:" << prop.multiProcessorCount
                      << std::endl;
            std::cout << "  CUDA 核心:"
                      << prop.multiProcessorCount *
                                 _ConvertSMVer2Cores(prop.major, prop.minor)
                      << std::endl;
            std::cout << "  CUDA 算力:" << prop.major << "." << prop.minor
                      << std::endl;
        }
    } else {
        std::cout << "CUDA 不可用,正在使用 CPU." << std::endl;
    }

    // 创建张量
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "随机张量:" << tensor << std::endl;

    // 基本运算
    torch::Tensor tensor_a = torch::tensor({1, 2, 3}, torch::kFloat32);
    torch::Tensor tensor_b = torch::tensor({4, 5, 6}, torch::kFloat32);
    torch::Tensor result = tensor_a + tensor_b;
    std::cout << "张量添加:" << result << std::endl;

    // 自动求导
    torch::Tensor x = torch::tensor({1.0, 2.0, 3.0}, torch::requires_grad());
    torch::Tensor y = x * 2;
    y.backward(torch::ones_like(y));
    std::cout << "坡度:" << x.grad() << std::endl;

    // 简单的线性回归模型
    struct Model : torch::nn::Module {
        Model() { fc = register_module("fc", torch::nn::Linear(3, 1)); }

        torch::Tensor forward(torch::Tensor x) { return fc->forward(x); }

        torch::nn::Linear fc{nullptr};
    };

    auto model = std::make_shared<Model>();
    torch::optim::SGD optimizer(model->parameters(),
                                torch::optim::SGDOptions(0.01));

    // 模拟一些数据
    torch::Tensor inputs = torch::rand({10, 3});
    torch::Tensor targets = torch::rand({10, 1});

    // 训练模型
    for (size_t epoch = 1; epoch <= 100; ++epoch) {
        optimizer.zero_grad();
        torch::Tensor outputs = model->forward(inputs);
        torch::Tensor loss = torch::mse_loss(outputs, targets);
        loss.backward();
        optimizer.step();

        if (epoch % 10 == 0) {
            std::cout << "Epoch [" << epoch
                      << "/100], Loss: " << loss.item<float>() << std::endl;
        }
    }

    // CUDA 测试
    if (torch::cuda::is_available()) {
        torch::Tensor tensor_cuda =
                torch::rand({2, 3}, torch::device(torch::kCUDA));
        std::cout << "CUDA 上的随机张量:" << tensor_cuda << std::endl;

        torch::Tensor result_cuda = tensor_cuda * 2;
        std::cout << "CUDA 上的张量乘法:" << result_cuda << std::endl;
    }

    return 0;
}

int _ConvertSMVer2Cores(int major, int minor) {
    switch ((major << 4) + minor) {
        case 0x30:  // Kepler
        case 0x32:  // Kepler
        case 0x35:  // Kepler
        case 0x37:  // Kepler
            return 192;

        case 0x50:  // Maxwell
        case 0x52:  // Maxwell
        case 0x53:  // Maxwell
            return 128;

        case 0x60:  // Pascal
            return 64;
        case 0x61:  // Pascal
        case 0x62:  // Pascal
            return 128;

        case 0x70:  // Volta
        case 0x72:  // Volta
        case 0x75:  // Turing
            return 64;

            // 新增架构支持
        case 0x80:  // Ampere
        case 0x86:  // Ampere
            return 128;
        case 0x87:  // Ampere
            return 64;

        case 0x90:  // Hopper
        case 0x92:  // Hopper
            return 128;

        case 0x89:       // 假设的新架构
            return 256;  // 示例值

        default:
            return -1;  // 未知架构
    }
}