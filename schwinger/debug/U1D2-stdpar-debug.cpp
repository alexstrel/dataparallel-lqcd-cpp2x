#include <core/common.h>
#include <fields/field_descriptor.h>
#include <fields/field.h>
//#include <dslash_factory.h>
//
#include <dslash_ref.h>
//
using Float   = float;

constexpr int ldim = 32;
constexpr int tdim = 32;

std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()s
std::uniform_real_distribution<Float> dis(0.f, 1.f);

void init_u1(auto &field){
   for (auto &i : field.Data()) i = std::polar(1.0f,dis(gen));	
}

void init_spinor(auto &field){
   for (auto &i : field.Data()) i = std::complex<Float>(dis(gen), dis(gen));
}

void check_spinor(auto &src1, auto &src2){

   Float tol = 1e-13;
   
   for (int i = 0; i < src1.GetLength(); i++){
//   for (int i = 0; i < 1024; i++){   
     if (norm(src1.Data()[i] - src2.Data()[i]) > tol) 
     {
       std::cout <<  src1.Data()[i] << " :: " << src2.Data()[i] << " " << i << std::endl;
     } 
   }
}

void fill(auto &src_field_accessor) {
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

void convert_lex2eo(auto &dst_field_accessor, const auto &src_field_accessor){

  const int mu = src_field_accessor.extent(2);
  const int V  = src_field_accessor.extent(0) * src_field_accessor.extent(1); 
  const int Vh = dst_field_accessor.extent(0) * dst_field_accessor.extent(1); 
  for(int p = 0; p < dst_field_accessor.extent(3); p++){    
    for(int i = 0; i < dst_field_accessor.extent(0); i++){
      for(int j = 0; j < dst_field_accessor.extent(1); j++){
        int parity_bit = j & 1; 	 
        int parity_offset = parity_bit == 0 ? p : (1-p);
#pragma unroll 
        for(int k = 0; k < mu; k++){
	  dst_field_accessor(i,j,k,p) = src_field_accessor(2*i+parity_offset,j,k);     
        }	       
      }
    }
  }
  return;
}

void convert_eo2lex(auto &dst_field_accessor, const auto &src_field_accessor){

  const int mu = src_field_accessor.extent(2);
  const int Vh = src_field_accessor.extent(0) * src_field_accessor.extent(1); 
  const int V  = dst_field_accessor.extent(0) * dst_field_accessor.extent(1); 
  for(int p = 0; p < src_field_accessor.extent(3); p++){    
    for(int i = 0; i < src_field_accessor.extent(0); i++){
      for(int j = 0; j < src_field_accessor.extent(1); j++){
        int parity_bit = j & 1; 	 
        int parity_offset = parity_bit == 0 ? p : (1-p);
#pragma unroll 
        for(int k = 0; k < mu; k++){
	  dst_field_accessor(2*i+parity_offset,j,k) = src_field_accessor(i,j,k,p);     
        }	       
      }
    }
  }
  return;
}

void print_range(auto &field, const int range){
   std::cout << "Print components for field : " << field.Data().data() << std::endl;

   auto print = [](const auto& e) { std::cout << "Element " << e << std::endl; };

   std::for_each(field.Data().begin(), field.Data().begin()+range, print);
}

void print_range(auto &field_accessor){

  const int mu = field_accessor.extent(2);
  {
    for(int j = 0; j < field_accessor.extent(1); j++){  
      for(int i = 0; i < field_accessor.extent(0); i++){
        for(int k = 0; k < mu; k++){
          auto x = field_accessor(i, j, k);
	  printf("Value at % d %d %d \tlex ( %f, %f)\t \n", i, j, k, x.real(), x.imag());
        }	       
      }
    }
  }
}

void print_eo_range(auto &field_accessor){
  const int mu = field_accessor.extent(2);
  for(int p = 0; p < field_accessor.extent(3); p++){  
    printf("parity %d \n", p);
    for(int j = 0; j < field_accessor.extent(1); j++){  
      for(int i = 0; i < field_accessor.extent(0); i++){
        for(int k = 0; k < mu; k++){
          auto x = field_accessor(i, j, k, p);
	  printf("Value at % d %d %d %d \teo ( %f, %f)\t \n", i, j, k, p, x.real(), x.imag());
        }	       
      }
    }
  }
}


void compare_lexeo(auto &dst_field_accessor, const auto &src_field_accessor, const double tol, const int parity){

  const int mu = src_field_accessor.extent(2);
  const int V  = src_field_accessor.extent(0) * src_field_accessor.extent(1); 
  const int Vh = dst_field_accessor.extent(0) * dst_field_accessor.extent(1); 
  {
    const int p = parity;    
    for(int i = 0; i < dst_field_accessor.extent(0); i++){
      for(int j = 0; j < dst_field_accessor.extent(1); j++){
        int parity_bit = j & 1; 	 
        int parity_offset = parity_bit == 0 ? p : (1-p);
#pragma unroll 
        for(int k = 0; k < mu; k++){
	  double diff_ = abs(dst_field_accessor(i,j,k,p) - src_field_accessor(2*i+parity_offset,j,k));     
	  if(diff_ > tol) 
	    std::cout << "Error found : diff = " << diff_ << " coords x=" << i << " y= " << j << "  parity field " << dst_field_accessor(i,j,k,p).real() << " (parity " << p << " ), lex ordered field " << src_field_accessor(2*i+parity_offset,j,k).real() << std::endl;
        }	       
      }
    }
  }
  return;
}

//--------------------------------------------------------------------------------
int main(int argc, char **argv)
{
  //
  using vector_tp = std::vector<std::complex<Float>>;

  const Float mass = 0.05;
  const Float r    = 1.0;
  
  constexpr int nParity = 2;
  //  
  const auto cs_param = SpinorFieldArgs<invalid_parity>{{ldim, tdim}, {0, 0, 0, 0}, FieldParity::InvalidFieldParity, FieldOrder::LexFieldOrder};  
  //
  auto src_spinor = create_field<vector_tp, decltype(cs_param)>(cs_param);
  auto dst_spinor = create_field<vector_tp, decltype(cs_param)>(cs_param);
  auto chk_spinor = create_field<vector_tp, decltype(cs_param)>(cs_param);  
  
  const auto gauge_param = GaugeFieldArgs<invalid_parity>{{ldim, tdim}, {0, 0}, FieldParity::InvalidFieldParity, FieldOrder::LexFieldOrder}; 
  //
  auto gauge = create_field<vector_tp, decltype(gauge_param)>(gauge_param);

  init_u1(gauge);
  init_spinor(src_spinor);
  
  // Setup dslash arguments:
  auto &&u_ref    = gauge.View();
  //
  auto &&in_ref    = src_spinor.View();
  auto &&out_ref   = dst_spinor.View();
  auto &&chk_ref   = chk_spinor.View();  
  
  auto out__ = out_ref.ParityAccessor(); 
  auto chk__ = chk_ref.ParityAccessor(); 
  auto in__  = in_ref.ParityAccessor();       
  //
  Dpsi(chk_ref, in_ref, u_ref, mass, r, ldim, tdim);
  
  //now lets do parity spinors:
  const auto eo_gauge_param = GaugeFieldArgs<nParity>{{ldim, tdim}, {0, 0}}; 
  const auto eo_cs_param    = SpinorFieldArgs<nParity>{{ldim, tdim}, {0, 0, 0, 0}};
  //
  auto eo_gauge  = create_field<vector_tp, decltype(eo_gauge_param)>(eo_gauge_param);  
  //
  auto eo_src_spinor = create_field<vector_tp, decltype(eo_cs_param)>(eo_cs_param);    
  auto eo_dst_spinor = create_field<vector_tp, decltype(eo_cs_param)>(eo_cs_param);
  //
  auto&& eo_src_spinor_ref = eo_src_spinor.View();
  auto&& eo_dst_spinor_ref = eo_dst_spinor.View();    
  //
  auto eo_view = eo_src_spinor.Accessor();
  //auto eo_view = eo_src_spinor_ref.Accessor();  
  auto in_view = in_ref.ParityAccessor();
  //
  convert_lex2eo(eo_view, in_view);         
  //
  auto [e_src_spinor_ref, o_src_spinor_ref] = eo_src_spinor.EODecompose();    
  //
  auto &&eo_u_ref = eo_gauge.View(); 
  //
  auto eo_u  = eo_u_ref.Accessor();   
  auto lex_u = u_ref.ParityAccessor();     
  //
  convert_lex2eo(eo_u, lex_u);     
  //  
  auto&& o_dst_spinor_ref = eo_dst_spinor.Odd();
  //
  Dpsi_parity(o_dst_spinor_ref, e_src_spinor_ref, o_src_spinor_ref, eo_u_ref, mass, r, ldim / 2, tdim, 1);  
  //
  auto chk2__ = o_dst_spinor_ref.ParityAccessor(); 
  //
  auto eo_check  = eo_dst_spinor_ref.Accessor();
  auto lex_check = chk_ref.ParityAccessor();
  //  
  compare_lexeo(eo_check, lex_check, 1e-8, 1);
  // initialize the data
  bool verbose = true;
  
  if (verbose > 0) {
    std::cout << "Number of sites = " << ldim << " x " << tdim << "." << std::endl;
    std::cout << std::flush;
  }

  return 0;
}
