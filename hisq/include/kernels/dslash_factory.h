#pragma once
#include <optional>
#include <typeinfo>
//
#include <kernels/staggered_dslash.h>
#include <core/cartesian_product.hpp>
//
#include <fields/field.h>
#include <fields/field_accessor.h>

#include <typeinfo>

// Custom concepts for both single and block spinors:
template<typename T> concept ParitySpinorField = GenericStaggeredParitySpinorFieldTp<T> or GenericBlockStaggeredParitySpinorFieldTp<T>;

template<typename T> concept FullSpinorField   = GenericStaggeredFullSpinorFieldTp<T>   or GenericBlockStaggeredFullSpinorFieldTp<T>;

template<typename T> concept GaugeField = GenericGaugeFieldTp<T> and requires { requires (T::Nparity() == 2); };

//DslashTransform
template<typename KernelArgs, template <typename Args> class Kernel>
class DslashTransform{
  private:
    std::unique_ptr<Kernel<KernelArgs>> dslash_kernel_ptr;
     
  public:
    using kernel_data_tp = typename std::remove_cvref_t<KernelArgs>::gauge_data_tp;
    
    static constexpr std::size_t bSize  = std::remove_cvref_t<KernelArgs>::bSize;

    DslashTransform(const KernelArgs &args) : dslash_kernel_ptr(new Kernel<KernelArgs>(args)) {}
    
    KernelArgs& ExportKernelArgs() const { return dslash_kernel_ptr->args; }
    
    template<bool dagger>
    inline void launch_dslash(GenericStaggeredSpinorFieldViewTp auto &out_view, const GenericStaggeredSpinorFieldViewTp auto &in_view, const GenericStaggeredSpinorFieldViewTp auto &aux_view, auto&& post_transformer, const FieldParity parity, const auto ids) {
      
      auto DslashKernel = [=, &dslash_kernel   = *dslash_kernel_ptr] (const auto coords) { 
                            //
                            dslash_kernel.template apply<dagger>(out_view, in_view, aux_view, post_transformer, coords, parity); 
                          };
      //
      std::for_each(std::execution::par_unseq,
                    ids.begin(),
                    ids.end(),
                    DslashKernel);    
    } 
    
    template<bool dagger>
    inline void launch_dslash(GenericStaggeredSpinorFieldViewTp auto &out_view, const GenericStaggeredSpinorFieldViewTp auto &in_view, const FieldParity parity, const auto ids) {
      
      auto DslashKernel = [=, &dslash_kernel   = *dslash_kernel_ptr] (const auto coords) { 
                            //
                            dslash_kernel.template apply<dagger>(out_view, in_view, coords, parity); 
                          };
      //
      std::for_each(std::execution::par_unseq,
                    ids.begin(),
                    ids.end(),
                    DslashKernel);    
    }    

    template<bool dagger> 
    void operator()(GenericStaggeredParitySpinorFieldTp auto &out, const GenericStaggeredParitySpinorFieldTp auto &in, const GenericStaggeredParitySpinorFieldTp auto &aux, auto&& post_transformer, const FieldParity parity){
      // Setup exe domain
      auto ids = impl::cartesian_4d_view(out.GetCBDims());

      // Create views of the arguments (possible launch overhead!): 
      auto&& out_view       = out.View();
      const auto&& in_view  = in.View();
      const auto&& aux_view = aux.View();         
      
      // Launch the kernels  
      launch_dslash<dagger>(out_view, in_view, aux_view, post_transformer, parity, ids);
    }
    
    template<bool dagger>
    void operator()(GenericBlockStaggeredSpinorFieldTp auto &out_block_spinor, GenericBlockStaggeredSpinorFieldTp auto &in_block_spinor, GenericBlockStaggeredSpinorFieldTp auto &aux_block_spinor, auto&& post_transformer, const FieldParity parity){ 
      //Setup exe domain :
      auto ids = impl::cartesian_4d_view(out_block_spinor.GetCBDims());      

      //First, we need to convert to views all components in the block
      auto &&out_block_spinor_view    = out_block_spinor.ConvertToView();
      auto &&in_block_spinor_view     = in_block_spinor.ConvertToView();       
      auto &&aux_block_spinor_view    = aux_block_spinor.ConvertToView();               

      //Second, create a view for the whole block spinor container:
      auto &&out_view    = out_block_spinor_view.BlockView();
      auto &&in_view     = in_block_spinor_view.BlockView(); 
      auto &&aux_view    = aux_block_spinor_view.BlockView();         
        
      //Finally, launch the kernel:
      launch_dslash<dagger>(out_view, in_view, aux_view, post_transformer, parity, ids);  
    }    
};

template<typename KernelArgs, template <typename Args> class Kernel, typename TransformParams>
class Mat : public DslashTransform<KernelArgs, Kernel> {
  private:
    const TransformParams param;
    
    const FieldParity parity;
 
  public:

    Mat(const KernelArgs &args, const TransformParams &param, const FieldParity parity = FieldParity::InvalidFieldParity) : DslashTransform<KernelArgs, Kernel>(args), param(param), parity(parity) {}

    template<bool dagger = false>
    void DslashXpay(ParitySpinorField auto &out, const ParitySpinorField auto &in, const ParitySpinorField auto &aux){
      // Check all arguments!      
      
      using SpinorTp = typename std::remove_cvref_t<decltype(out[0])>; 
      //
      constexpr int nDoF   = SpinorTp::Type() == FieldType::StaggeredSpinorFieldType? SpinorTp::Ncolor() : SpinorTp::Ncolor() * SpinorTp::Nspin();
      
      constexpr int bsize = DslashTransform<KernelArgs, Kernel>::bSize;
      
      const auto c = static_cast<DslashTransform<KernelArgs, Kernel>::kernel_data_tp>(2.0*param.M);
      
      auto transformer = [=](const auto &x, auto &y) {
        //
        const auto &&x_ = x.flat_cview();
        auto &&y_ = y.flat_view();        
        //
#pragma unroll              
        for(int n = 0; n < nDoF; n++ ) {
#pragma unroll              
          for(int b = 0; b < bsize; b++) {
            y_(n, b) = (c*x_(n, b)-y_(n, b));
          }
        }
      };      
      //
      DslashTransform<KernelArgs, Kernel>::template operator()<dagger>(out, in,  aux, transformer, parity);
    }

    template<bool dagger = false>
    void operator()(ParitySpinorField auto &out, const ParitySpinorField auto &in){
      // Check all arguments!      
      DslashXpay<dagger>(out, in, in);
    }

    template<bool dagger = false>
    void operator()(FullSpinorField auto &out, FullSpinorField auto &in){//FIXME: in argument must be constant
      // Check all arguments!
      
      using SpinorTp     = typename std::remove_cvref_t<decltype(out[0])>; 
      //
      constexpr int nDoF   = SpinorTp::Type() == FieldType::StaggeredSpinorFieldType? SpinorTp::Ncolor() : SpinorTp::Ncolor() * SpinorTp::Nspin();
      //
      constexpr int bsize  = DslashTransform<KernelArgs, Kernel>::bSize;      

      const auto c = static_cast<DslashTransform<KernelArgs, Kernel>::kernel_data_tp>(2*param.M);

      auto [even_in,   odd_in] = in.EODecompose();
      auto [even_out, odd_out] = out.EODecompose();
      //
      {            
        auto transformer = [=](const auto &x, auto &y) {
          //
          const auto &&x_ = x.flat_cview();
          auto &&y_ = y.flat_view();        
          //
#pragma unroll              
          for(int n = 0; n < nDoF; n++ ) {
#pragma unroll              
            for(int b = 0; b < bsize; b++) {
              y_(n, b) = (c*x_(n, b)-y_(n, b));
            }
          }
        }; 
        //
        DslashTransform<KernelArgs, Kernel>::template operator()<dagger>(even_out, odd_in,  even_in, transformer, FieldParity::EvenFieldParity);
        DslashTransform<KernelArgs, Kernel>::template operator()<dagger>(odd_out,  even_in, odd_in,  transformer, FieldParity::OddFieldParity);       
      }
    }    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    
};


