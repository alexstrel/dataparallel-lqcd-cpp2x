#include <fields/field.h>
#include <fields/block_field.h>
#include <kernels/dslash_factory.h>
#include <core/memory.h> 
#include <numbers>

#include <utils/gauge_utils.h>

//
using Float = double;

void init_su3(auto &field){
   for (auto &i : field.Data()) i = std::polar(static_cast<Float>(1.f),dis(gen));
}

void init_spinor(auto &field){
   for (auto &i : field.Data()) i = dis(gen); //std::complex<Float>(1.f, 0.f);
}

template<GenericStaggeredSpinorFieldTp field_tp>
void print_range(field_tp &field, const int range){
   std::cout << "Print components for field : " << field.Data().data() << std::endl;

   auto print = [](const auto& e) { std::cout << "Element " << e << std::endl; };

   std::for_each(field.Data().begin(), field.Data().begin()+range, print);
}

void check_field(const auto &dst_field_accessor, const auto &src_field_accessor, const double tol){

  const int mu = src_field_accessor.extent(4);
  const int V  = src_field_accessor.extent(0) * src_field_accessor.extent(1)*src_field_accessor.extent(2) * src_field_accessor.extent(3); 
  {
    for(int i = 0; i < dst_field_accessor.extent(0); i++){
      for(int j = 0; j < dst_field_accessor.extent(1); j++){
        for(int k = 0; k < dst_field_accessor.extent(2); k++){      
          for(int l = 0; l < dst_field_accessor.extent(3); l++){              
#pragma unroll 
            for(int s = 0; s < mu; s++){
	      double diff_ = abs(dst_field_accessor(i,j,k,l,s) - src_field_accessor(i,j,k,l,s));     
	      if(diff_ > tol) 
	        std::cout << "Error found : diff = " << diff_ << " coords x=" << i << " y= " << j << " z= " << k << " t= " << l << "  check field " << dst_field_accessor(i,j,k,l,s).real() << " orig field " << src_field_accessor(i,j,k,l,s).real() << std::endl;
	    }
	  }
        }	       
      }
    }
  }
  return;
}

//
#include <dslash_pmr_test.h>

//--------------------------------------------------------------------------------
int main(int argc, char **argv)
{
  //
  constexpr int X = 32;
  constexpr int T = 32;

  const Float mass = 0.05;

  DslashParam<Float> dslash_param{mass};

  const int niter = 100;
  
  std::array dims = {X, X, X, T};
  
  //run_pmr_dslash_test(dslash_param, dims, niter, 0);
  //
  constexpr int  N = 8;  
  //
  run_mrhs_pmr_dslash_test<N>(dslash_param, dims, niter, 0);

  // initialize the data
  bool verbose = true;
  
  if (verbose > 0) {
    std::cout << "Number of sites = " << X << " x " << T << "." << std::endl;
    std::cout << std::flush;
  }

  return 0;
}
