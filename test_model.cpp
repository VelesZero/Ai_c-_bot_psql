#include <iostream>
#include <torch/torch.h>

int main() {
    std::cout << "Test model is working!" << std::endl;
    std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;
    
    // Test if CUDA is available
    std::cout << "CUDA available: " << (torch::cuda::is_available() ? "YES" : "NO") << std::endl;
    
    // Create a simple tensor
    auto tensor = torch::rand({2, 3});
    std::cout << "Random tensor:\n" << tensor << std::endl;
    
    return 0;
}
