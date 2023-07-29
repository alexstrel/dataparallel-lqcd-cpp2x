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

// Custom concept for both single and block spinors:
template<typename T> concept SpinorField = GenericStaggeredSpinorFieldTp<T> or GenericBlockStaggeredSpinorFieldTp<T>;

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
    
    inline void launch_dslash(GenericStaggeredSpinorFieldViewTp auto &out_view, const GenericStaggeredSpinorFieldViewTp auto &in_view, const GenericStaggeredSpinorFieldViewTp auto &aux_view, auto&& post_transformer, const FieldParity parity, const auto ids) {
      
      auto DslashKernel = [=, &dslash_kernel   = *dslash_kernel_ptr] (const auto coords) { 
                            //
                            dslash_kernel.template apply(out_view, in_view, aux_view, post_transformer, coords, parity); 
                          };
      //
      std::for_each(std::execution::par_unseq,
                    ids.begin(),
                    ids.end(),
                    DslashKernel);    
    } 
    
    inline void launch_dslash(GenericStaggeredSpinorFieldViewTp auto &out_view, const GenericStaggeredSpinorFieldViewTp auto &in_view, const FieldParity parity, const auto ids) {
      
      auto DslashKernel = [=, &dslash_kernel   = *dslash_kernel_ptr] (const auto coords) { 
                            //
                            dslash_kernel.template apply(out_view, in_view, coords, parity); 
                          };
      //
      std::for_each(std::execution::par_unseq,
                    ids.begin(),
                    ids.end(),
                    DslashKernel);    
    }    
 
    void operator()(GenericStaggeredSpinorFieldTp auto &out, const GenericStaggeredSpinorFieldTp auto &in, const GenericStaggeredSpinorFieldTp auto &aux, auto&& post_transformer, const FieldParity parity){
      
      if ( in.GetFieldOrder() != FieldOrder::EOFieldOrder and in.GetFieldSubset() != FieldSiteSubset::ParitySiteSubset ) { 
        std::cerr << "Only parity field is allowed." << std::endl; 
        std::quick_exit( EXIT_FAILURE );  
      }    

      using spinor_tp    = typename std::remove_cvref_t<decltype(in)>;
      using container_tp = spinor_tp::container_tp;
      
      //Setup exe domain
      const auto [Nx, Ny, Nz, Nt] = out.GetCBDims(); //Get CB dimensions
      
      auto X = std::views::iota(0, Nx);
      auto Y = std::views::iota(0, Ny);
      auto Z = std::views::iota(0, Nz);
      auto T = std::views::iota(0, Nt);      

      auto ids = std::views::cartesian_product(T, Z, Y, X);//T is the slowest index, X is the fastest
      
      if constexpr (is_allocator_aware_type<container_tp> or is_pmr_allocator_aware_type<container_tp>) {
        auto&& out_view       = out.View();
        const auto&& in_view  = in.View();
        const auto&& aux_view = aux.View();         
        
        launch_dslash(out_view, in_view, aux_view, post_transformer, parity, ids);
      } else {
        launch_dslash(out, in, aux, post_transformer, parity, ids);  
      }       
    }
    
    void operator()(GenericBlockStaggeredSpinorFieldTp auto &out_block_spinor, GenericBlockStaggeredSpinorFieldTp auto &in_block_spinor, GenericBlockStaggeredSpinorFieldTp auto &aux_block_spinor, auto&& post_transformer, const FieldParity parity){ 
      //   
      assert(in_block_spinor.GetFieldOrder() == FieldOrder::EOFieldOrder and in_block_spinor.GetFieldSubset() == FieldSiteSubset::ParitySiteSubset);
      
      using block_spinor_tp        = typename std::remove_cvref_t<decltype(in_block_spinor)>;
      using component_container_tp = block_spinor_tp::container_tp;      
      
      //Setup exe domain
      const auto [Nx, Ny, Nz, Nt] = out.GetCBDims(); //Get CB dimensions
      
      auto X = std::views::iota(0, Nx);
      auto Y = std::views::iota(0, Ny);
      auto Z = std::views::iota(0, Nz);
      auto T = std::views::iota(0, Nt);      

      auto ids = std::views::cartesian_product(T, Z, Y, X);//T is the slowest index, X is the fastest
                    
     if constexpr (is_allocator_aware_type<component_container_tp> or is_pmr_allocator_aware_type<component_container_tp>) {
        //First, we need to convert to views all components in the block
        auto &&out_block_spinor_view    = out_block_spinor.ConvertToView();
        auto &&in_block_spinor_view     = in_block_spinor.ConvertToView();       
        auto &&aux_block_spinor_view    = aux_block_spinor.ConvertToView();               

        auto &&out_view    = out_block_spinor_view.BlockView();
        auto &&in_view     = in_block_spinor_view.BlockView(); 
        auto &&aux_view    = aux_block_spinor_view.BlockView();         
        
        launch_dslash(out_view, in_view, aux_view, post_transformer, parity, ids);  
      } else {
        auto &&out_view    = out_block_spinor.BlockView();
        auto &&in_view     = in_block_spinor.BlockView(); 
        auto &&aux_view  = aux_block_spinor.BlockView();         
      
        launch_dslash(out_view, in_view, aux_view, post_transformer, parity, ids);  
      }                    
    }    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    
};

template<typename KernelArgs, template <typename Args> class Kernel, typename TransformParams, bool do_normal = false>
class Mat : public DslashTransform<KernelArgs, Kernel> {
  private:
    const TransformParams param;
    
    const FieldParity parity;
 
  public:

    Mat(const KernelArgs &args, const TransformParams &param, const FieldParity parity = FieldParity::InvalidFieldParity, const bool dagger = false) : DslashTransform<KernelArgs, Kernel>(args), param(param), parity(parity) {}

    void operator()(SpinorField auto &out, const SpinorField auto &in, const SpinorField auto &aux){
      // Check all arguments!
      if(out.GetFieldSubset() != FieldSiteSubset::ParitySiteSubset) { 
        std::cerr << "Error: undefined parity.. exiting\n";
        std::quick_exit( EXIT_FAILURE );
      }       
      
      using SpinorTp = typename std::remove_cvref_t<decltype(out[0])>; 
      //
      constexpr int nDoF  = SpinorTp::Ncolor() * SpinorTp::Nspin();
      constexpr int bsize = DslashTransform<KernelArgs, Kernel>::bSize;
      
      const auto const1 = static_cast<DslashTransform<KernelArgs, Kernel>::kernel_data_tp>(param.M + 2.0*param.r);
      const auto const2 = static_cast<DslashTransform<KernelArgs, Kernel>::kernel_data_tp>(0.5);                
      
      auto transformer = [=](const auto &x, auto &y) {
        //
        const auto &&x_ = x.flat_cview();
        auto &&y_ = y.flat_view();        
        //
#pragma unroll              
        for(int n = 0; n < nDoF; n++ ) {
#pragma unroll              
          for(int b = 0; b < bsize; b++) {
            y_(n, b) = (const1*x_(n, b)-const2*y_(n, b));
          }
        }
      };      
      //
      DslashTransform<KernelArgs, Kernel>::operator()(out, in,  aux, transformer, parity);
    }
    
    void operator()(SpinorField auto &out, SpinorField auto &in){//FIXME: in argument must be constant
      // Check all arguments!
      if( out.GetFieldSubset() != FieldSiteSubset::FullSiteSubset ) { 
        std::cerr << "This operation is supported for full fields only...\n";
        std::quick_exit( EXIT_FAILURE );
      }  
      
      using SpinorTp     = typename std::remove_cvref_t<decltype(out[0])>; 
      //
      constexpr int nColor = SpinorTp::Ncolor();      
      constexpr int nSpin  = SpinorTp::Nspin();            
      constexpr int nDoF   = nColor * nSpin;
      //
      constexpr int bsize  = DslashTransform<KernelArgs, Kernel>::bSize;      

      const auto const1 = static_cast<DslashTransform<KernelArgs, Kernel>::kernel_data_tp>(param.M);
      const auto const2 = static_cast<DslashTransform<KernelArgs, Kernel>::kernel_data_tp>(0.5);                
      
      auto [even_in,   odd_in] = in.EODecompose();
      auto [even_out, odd_out] = out.EODecompose();
      //
      if constexpr (do_normal) {
        using pmr_container_tp = impl::pmr::vector<typename DslashTransform<KernelArgs, Kernel>::kernel_data_tp>;
        //
        auto tmp = create_field<decltype(in), pmr_container_tp>(in);
        
        auto [even_tmp, odd_tmp] = tmp.EODecompose();              
        
        auto transformer = [=](const auto &x, auto &y) {
          //
          const auto &&x_ = x.cview();
          auto &&y_ = y.view();        
          //
#pragma unroll              
          for(int c = 0; c < nColor; c++ ) {
#pragma unroll              
            for(int b = 0; b < bsize; b++) {
              y_(c, 0, b) =  (const1*x_(c, 0, b)-const2*y_(c, 0, b));
              y_(c, 1, b) = -(const1*x_(c, 1, b)-const2*y_(c, 1, b));              
            }
          }
        }; 

        //
        DslashTransform<KernelArgs, Kernel>::operator()(even_tmp, odd_in,  even_in, transformer, FieldParity::EvenFieldParity);
        DslashTransform<KernelArgs, Kernel>::operator()(odd_tmp,  even_in, odd_in,  transformer, FieldParity::OddFieldParity);      
        
        DslashTransform<KernelArgs, Kernel>::operator()(even_out, odd_tmp,  even_tmp, transformer, FieldParity::EvenFieldParity);
        DslashTransform<KernelArgs, Kernel>::operator()(odd_out,  even_tmp, odd_tmp,  transformer, FieldParity::OddFieldParity);      
      } else {            
        auto transformer = [=](const auto &x, auto &y) {
          //
          const auto &&x_ = x.flat_cview();
          auto &&y_ = y.flat_view();        
          //
#pragma unroll              
          for(int n = 0; n < nDoF; n++ ) {
#pragma unroll              
            for(int b = 0; b < bsize; b++) {
              y_(n, b) = (const1*x_(n, b)-const2*y_(n, b));
            }
          }
        }; 
        //
        DslashTransform<KernelArgs, Kernel>::operator()(even_out, odd_in,  even_in, transformer, FieldParity::EvenFieldParity);
        DslashTransform<KernelArgs, Kernel>::operator()(odd_out,  even_in, odd_in,  transformer, FieldParity::OddFieldParity);       
      }
    }    
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////    
};


