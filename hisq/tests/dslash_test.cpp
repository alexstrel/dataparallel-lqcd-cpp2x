#include <fields/field.h>
#include <fields/block_field.h>
#include <kernels/dslash_factory.h>

//
using Float   = float;
//
std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()s
std::uniform_real_distribution<Float> dis(0.f, 1.f);

void init_u1(auto &field){
   for (auto &i : field.Data()) i = std::polar(static_cast<Float>(1.f),dis(gen));	
}

void init_spinor(auto &field){
   for (auto &i : field.Data()) i = dis(gen); //std::complex<Float>(1.f, 0.f);
}

void fill(auto &field_accessor) {
  const int mu = field_accessor.extent(2);
  for(int i = 0; i < field_accessor.extent(0); i++){
    for(int j = 0; j < field_accessor.extent(1); j++){	 
#pragma unroll 
      for(int k = 0; k < mu; k++){
	 field_accessor(i,j,k) = std::complex<Float>(1.0, 0.0);     
      }	       
    }
  }
}

//template<FieldTp field_tp>
template<GenericSpinorFieldTp field_tp>
void print_range(field_tp &field, const int range){
   std::cout << "Print components for field : " << field.Data().data() << std::endl;

   auto print = [](const auto& e) { std::cout << "Element " << e << std::endl; };

   std::for_each(field.Data().begin(), field.Data().begin()+range, print);
}

template<int N, bool to_eo = true>
void convert_field(auto &dst_field, auto &src_field){
   const auto [Nx, Ny] = src_field.GetDims();

   auto X = std::views::iota(0, Nx);
   auto Y = std::views::iota(0, Ny);

   auto idx = std::views::cartesian_product(Y, X);//Y is the slowest index, X is the fastest

   auto &&dst = dst_field.View();
   auto &&src = src_field.View(); 

   std::for_each(std::execution::par_unseq,
                 idx.begin(),
                 idx.end(), [=](const auto i) {
		      auto [y, x] = i;
		      const int parity = (x + y) & 1;
		      // 
		      auto dstU = dst.Accessor();
		      auto srcU = src.template ParityAccessor();
#pragma unroll
		      for(int j = 0; j < N; j++){
		        if constexpr (to_eo) {  
		          dstU(x/2,y,j,parity) = srcU(x,y,j);
			} else {
			  srcU(x,y,j) = dstU(x/2,y,j,parity);
	                }		
		      }
		   });

}

void check_field(const auto &dst_field_accessor, const auto &src_field_accessor, const double tol){

  const int mu = src_field_accessor.extent(2);
  const int V  = src_field_accessor.extent(0) * src_field_accessor.extent(1); 
  {
    for(int i = 0; i < dst_field_accessor.extent(0); i++){
      for(int j = 0; j < dst_field_accessor.extent(1); j++){
#pragma unroll 
        for(int k = 0; k < mu; k++){
	  double diff_ = abs(dst_field_accessor(i,j,k) - src_field_accessor(i,j,k));     
	  if(diff_ > tol) 
	    std::cout << "Error found : diff = " << diff_ << " coords x=" << i << " y= " << j << "  check field " << dst_field_accessor(i,j,k).real() << " orig field " << src_field_accessor(i,j,k).real() << std::endl;
        }	       
      }
    }
  }
  return;
}

#include <dslash_test.h>

//--------------------------------------------------------------------------------
int main(int argc, char **argv)
{
  //
  constexpr int X = 2048;
  constexpr int T = 2048;

  const Float mass = 0.05;
  const Float r    = 1.0;

  DslashParam<Float> dslash_param{mass, r};

  const int niter = 1000;
  //
  run_dslash_test(dslash_param, X, T, niter);
  //
  constexpr int  N = 8; 
  //
  run_mrhs_dslash_test<N>(dslash_param, X, T, niter);

  // initialize the data
  bool verbose = true;
  
  if (verbose > 0) {
    std::cout << "Number of sites = " << X << " x " << T << "." << std::endl;
    std::cout << std::flush;
  }

  return 0;
}
