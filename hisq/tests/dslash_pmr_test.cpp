#include <fields/field.h>
#include <fields/block_field.h>
#include <kernels/dslash_factory.h>
#include <core/memory.h> 

//
using Float = double;
//
std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()s
std::uniform_real_distribution<Float> dis(0.f, 1.f);

void init_su3(auto &field){
   for (auto &i : field.Data()) i = std::polar(static_cast<Float>(1.f),dis(gen));	
}

void init_spinor(auto &field){
   for (auto &i : field.Data()) i = dis(gen); //std::complex<Float>(1.f, 0.f);
}


//template<FieldTp field_tp>
template<GenericStaggeredSpinorFieldTp field_tp>
void print_range(field_tp &field, const int range){
   std::cout << "Print components for field : " << field.Data().data() << std::endl;

   auto print = [](const auto& e) { std::cout << "Element " << e << std::endl; };

   std::for_each(field.Data().begin(), field.Data().begin()+range, print);
}

#include <dslash_pmr_test.h>

//--------------------------------------------------------------------------------
int main(int argc, char **argv)
{
  //
  constexpr int X = 16;
  constexpr int T = 16;

  const Float mass = 0.05;

  DslashParam<Float> dslash_param{mass};

  const int niter = 1;
  
  std::array dims = {X, X, X, T};
  
  run_pmr_dslash_test(dslash_param, dims, niter);
  //
  constexpr int  N = 8;  
  //
  //run_mrhs_pmr_dslash_test<N>(dslash_param, dims, niter);

  // initialize the data
  bool verbose = true;
  
  if (verbose > 0) {
    std::cout << "Number of sites = " << X << " x " << T << "." << std::endl;
    std::cout << std::flush;
  }

  return 0;
}
