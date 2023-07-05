#include <field.h>
#include <block_field.h>
#include <dslash_factory.h>
//
using Float   = float;

//constexpr int D = 2;
//constexpr int N = 2;

constexpr int ldim = 1024;
constexpr int tdim = 1024;

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()s
std::uniform_real_distribution<Float> dis(0.f, 1.f);

void init_u1(auto &field){
   for (auto &i : field.Data()) i = std::polar(1.0f,dis(gen));	
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
		      auto dstU = dst.ExtAccessor();
		      auto srcU = src.template Accessor<to_eo>();
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

template<int nDir, int nSpin>
void run_simple_dslash(auto params) {
  const auto sf_args = SpinorFieldArgs<nSpin>{ldim, tdim};
  //
  auto src_spinor = create_field<vector_tp, decltype(sf_args)>(sf_args);
  auto dst_spinor = create_field<vector_tp, decltype(sf_args)>(sf_args);
  auto chk_spinor = create_field<vector_tp, decltype(sf_args)>(sf_args);
  //
  const auto gf_args    = GaugeFieldArgs<nDir>{ldim, tdim};
  
}

//--------------------------------------------------------------------------------
int main(int argc, char **argv)
{
  //
  using vector_tp = std::vector<std::complex<Float>>;
  //
  constexpr int nDir  = 2;  
  constexpr int nSpin = 2;  

  const Float mass = 0.05;
  const Float r    = 1.0;

  DslashParam<Float> dslash_param{mass, r};

  // allocate and initialize the working lattices, matrices, and vectors
  //
  const auto sf_args    = SpinorFieldArgs<nSpin>{ldim, tdim};
  //
  const auto eo_sf_args = SpinorFieldArgs<nSpin>{ldim/2, tdim, FieldOrder::EOFieldOrder};
  //
  auto src_spinor = create_field<vector_tp, decltype(sf_args)>(sf_args);
  auto dst_spinor = create_field<vector_tp, decltype(sf_args)>(sf_args);
  auto chk_spinor = create_field<vector_tp, decltype(sf_args)>(sf_args);

  auto eo_src_spinor = create_field<vector_tp, decltype(eo_sf_args)>(eo_sf_args);
  auto eo_dst_spinor = create_field<vector_tp, decltype(eo_sf_args)>(eo_sf_args);

  const auto gf_args    = GaugeFieldArgs<nDir>{ldim, tdim};
  //
  const auto eo_gf_args = GaugeFieldArgs<nDir>{ldim/2, tdim, FieldOrder::EOFieldOrder}; 
  //
  auto gauge      = create_field<vector_tp, decltype(gf_args)>(gf_args);
  //
  auto eo_gauge   = create_field<vector_tp, decltype(eo_gf_args)>(eo_gf_args);

  init_u1(gauge);
  init_spinor(src_spinor);
 
  convert_field<nDir>(eo_gauge, gauge);
  convert_field<nSpin>(eo_src_spinor, src_spinor);

  print_range<decltype(src_spinor)>(src_spinor, 8);
  print_range<decltype(eo_src_spinor)>(eo_src_spinor, 16);  
  
  // Setup dslash arguments:
  auto &&u_ref        = gauge.View();
  using gauge_tp      = decltype(gauge.View());
  using spinor_ref_tp = decltype(src_spinor.View());

  constexpr std::size_t nspin = 2;

  std::unique_ptr<DslashArgs<gauge_tp, nspin>> dslash_args_ptr(new DslashArgs{u_ref});

  auto &dslash_args = *dslash_args_ptr;

  // Create dslash matrix
  auto mat = Mat<decltype(dslash_args), Dslash, decltype(dslash_param)>{dslash_args, dslash_param};  

  const int niter = 1000;
  
  const auto scale1 = mass + static_cast<Float>(2.0)*r;
  const auto scale2 = static_cast<Float>(0.5);

  auto transformer = [=](const auto &x, const auto &y) {return (scale1*x-scale2*y);};  

  for(int i = 0; i < niter; i++) {
    // Apply dslash	  
    mat(dst_spinor, src_spinor, transformer, FieldOrder::LexFieldOrder);
  }
  /////////
  
  
  
  
  
  
  
  
  
  
  
  
  
  auto &&eo_u_ref        = eo_gauge.View();
  using eo_gauge_tp      = decltype(eo_gauge.View());
  
  
  std::unique_ptr<DslashArgs<eo_gauge_tp, nspin>> eo_dslash_args_ptr(new DslashArgs{eo_u_ref});

  auto &eo_dslash_args = *eo_dslash_args_ptr;

  // Create dslash matrix
  auto eo_mat = Mat<decltype(eo_dslash_args), Dslash, decltype(dslash_param)>{eo_dslash_args, dslash_param};  


  for(int i = 0; i < niter; i++) {
    // Apply dslash	  
    eo_mat(eo_dst_spinor, eo_src_spinor, transformer, FieldOrder::EOFieldOrder);
  }  
  
  
  
  
  
  // Block spinors:
  using spinor_t     = decltype(src_spinor);
  //
  auto src_block_spinor = BlockSpinor< spinor_t, decltype(sf_args) >{sf_args, 2};
  //
  for (int i = 0; i < src_block_spinor.Size(); i++) init_spinor( src_block_spinor.v[i] );
  
  auto dst_block_spinor = BlockSpinor< spinor_t, decltype(sf_args) >{sf_args, 2};

  for(int i = 0; i < niter; i++) {
    mat(dst_block_spinor, src_block_spinor);
  }

  // initialize the data
  bool verbose = true;
  
  if (verbose > 0) {
    std::cout << "Number of sites = " << ldim << " x " << tdim << "." << std::endl;
    std::cout << std::flush;
  }

  return 0;
}
