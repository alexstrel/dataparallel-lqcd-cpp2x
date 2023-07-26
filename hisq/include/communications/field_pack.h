#pragma once

#include <communications/dslash.h>
#include <core/cartesian_product.hpp>

template<typename KernelArgs, template <typename Args> class Kernel>
class Pack{
  private:
    std::unique_ptr<Kernel<KernelArgs>> pack_kernel_ptr;

  public:

    Pack(const KernelArgs &args) : pack_kernel_ptr(new Kernel<KernelArgs>(args)) {}

    void operator()(SpinorFieldTp auto &in, auto&& transformer, const FieldOrder order = FieldOrder::EOFieldOrder){
      if ( order == FieldOrder::EOFieldOrder ) {
        
        const auto [Nx, Ny] = in.GetCBDims(); 
        
        auto X = std::views::iota(0, Nx);
        auto Y = std::views::iota(0, Ny);
      } else {
        auto &&in_  = in.View();
      }
    }

    void operator()(GaugeFieldTp auto &in, auto&& transformer, const FieldOrder order = FieldOrder::EOFieldOrder){
      if ( order == FieldOrder::EOFieldOrder ) {

        const auto [Nx, Ny] = in.GetCBDims(); 

        auto X = std::views::iota(0, Nx);
        auto Y = std::views::iota(0, Ny);
      } else {
      }
    }    
};


