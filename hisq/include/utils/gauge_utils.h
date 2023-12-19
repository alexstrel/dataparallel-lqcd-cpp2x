#include <core/memory.h> 
#include <numbers>
#include <algorithm>
#include <core/cartesian_product.hpp>
//
#include <fields/field.h>
#include <fields/field_accessor.h>

//
using Float = double;
//
std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()s
std::uniform_real_distribution<Float> dis(0.f, 1.f);

void applyGaugeFieldScaling(auto &gauge_field, const FloatTp auto anisotropy) {

  using data_tp = typename std::remove_cvref_t<decltype(gauge_field)>::data_tp;

  constexpr bool is_constant = false;
  constexpr int  bSize       = 1;

  auto policy = std::execution::par_unseq;
 
  std::for_each(policy, gauge_field.Begin(), gauge_field.End(), [=](auto &v) {  v *= 1.0 / anisotropy; } );  	

  // Apply boundary conditions to temporal links
  if (gauge_field.GetTBoundary() == FieldBoundary::AntiPeriodicBC) {
  
    auto &&gauge_ref = gauge_field.View(); 
    
    using LinkAccessor = FieldAccessor<decltype(gauge_field.View()), is_constant, bSize>;//non-constant links, bsize = 1 (always!)   
    using Link         = LinkAccessor::LinkTp;     
 
    const auto [Nx, Ny, Nz, Nt] = gauge_field.GetCBDims();
  
    auto ids = impl::cartesian_4d_restricted_view({0, 0, 0, (Nt-1)}, {Nx, Ny, Nz, Nt}); 
    
    auto links = LinkAccessor(gauge_ref); 

    std::for_each(policy,
                  ids.begin(),
                  ids.end(),
                  [=] (const auto cartesian_idx) { 
                  
                    constexpr int dir = 3;
                                      
                    auto X  = impl::convert_coords<4>(cartesian_idx);
                    //
                    Link Ue = links(X, dir, 0);
                    Link Uo = links(X, dir, 1); 
                    //
                    Ue *= -1.0;
                    Uo *= -1.0;                                                                                                
                  });    
    
  }

  return;
}

void constructUnitGaugeField(auto &gauge_field, const FloatTp auto anisotropy) {

  using data_tp = typename std::remove_cvref_t<decltype(gauge_field)>::data_tp;

  constexpr bool is_constant = false;
  constexpr int  bSize       = 1;

  auto policy = std::execution::par_unseq;

  auto &&gauge_ref = gauge_field.View(); 
    
  using LinkAccessor = FieldAccessor<decltype(gauge_field.View()),  is_constant, bSize>;//non-constant links   
  using Link         = LinkAccessor::LinkTp;     
  
  auto ids = impl::cartesian_4d_view(gauge_field.GetCBDims());

  auto links = LinkAccessor(gauge_ref); 
  
  std::for_each(policy,
                ids.begin(),
                ids.end(),
                [=] (const auto cartesian_idx) { 
                                      
                  auto X  = impl::convert_coords<4>(cartesian_idx);
#pragma unroll                  
                  for(int d = 0; d < 4; d++) {
                    //
                    Link Ue = links(X, d, 0);
                    Link Uo = links(X, d, 1); 
                    //
                    Ue.unit();
                    Uo.unit();               
                  }                                                                                 
                });      

  applyGaugeFieldScaling(gauge_field, anisotropy);
}

// 0 -> STAGGERED_DSLASH,
// 1 -> ASQTAD_DSLASH
template <int dslash_type = 0>
void applyGaugeFieldScaling_long(auto &gauge_field, const FloatTp auto tadpole_coeff) {

  using data_tp = typename std::remove_cvref_t<decltype(gauge_field)>::data_tp;
  using real_tp = typename data_tp::value_type;


  constexpr bool is_constant = false;
  constexpr int  bSize       = 1;
  //
  auto policy = std::execution::par_unseq;

  // rescale long links by the appropriate coefficient
  if (dslash_type == 1) {//ASQTAD_DSLASH

    std::for_each(policy, gauge_field.Begin(), gauge_field.End(), [=](auto &v) { v *= 1.0 / (-24 * tadpole_coeff * tadpole_coeff); } );       
  }

  auto &&gauge_ref = gauge_field.View(); 
    
  using LinkAccessor = FieldAccessor<decltype(gauge_field.View()), is_constant, bSize>;//non-constant links   
  using Link         = LinkAccessor::LinkTp;     
  
  const auto [Nx, Ny, Nz, Nt] = gauge_field.GetCBDims();  
  
  auto ids = impl::cartesian_4d_view({Nx, Ny, Nz, Nt}); 

  auto links = LinkAccessor(gauge_ref);

  std::for_each(policy,
                ids.begin(),
                ids.end(),
                [=] (const auto cartesian_idx) { 
                                      
                  auto X  = impl::convert_coords<4>(cartesian_idx);
                  const auto [x,y,z,t] = X;//decompose the array of coords
                  //
#pragma unroll                  
                  for (int d = 0; d < 3; d++) {             
                    //     
                    Link Ue = links(X, d, 0);
                    Link Uo = links(X, d, 1); 
                    //
                    real_tp sign = 1.0;

                    if ((d == 0) and (t % 2 == 1))           sign = -1.0; 

                    if ((d == 1) and ((t + x) % 2 == 1))     sign = -1.0; 
                    
                    if ((d == 2) and ((t + x + y) % 2 == 1)) sign = -1.0; 
                              
                    Ue *= sign;
                    Uo *= sign;                              
                  }
                });    

  // Apply boundary conditions to temporal links
  if (gauge_field.GetTBoundary() == FieldBoundary::AntiPeriodicBC) {
  
    auto restricted_ids = impl::cartesian_4d_restricted_view({0, 0, 0, dslash_type == 0 ? (Nt-1) : (Nt-3)}, {Nx, Ny, Nz, Nt});  

    std::for_each(policy,
                  restricted_ids.begin(),
                  restricted_ids.end(),
                  [=] (const auto restricted_cartesian_idx) { 
                  
                    constexpr int dir = 3;
                                      
                    auto X  = impl::convert_coords<4>(restricted_cartesian_idx);
                    //
                    Link Ue = links(X, dir, 0);
                    Link Uo = links(X, dir, 1); 
                    //
                    Ue *= -1.0;
                    Uo *= -1.0;                                                                                                
                  });    
  
  }
}

// normalize a row of the link, with row idx row_idx:
template<int ncol>
static void normalize(MDViewTp auto &link, const int row_idx) {
  constexpr int bSize = 1;//otherwise would need loop over bsize
  
  double sum = 0.0;
#pragma unroll  
  for (int i = 0; i < ncol; i++) {
#pragma unroll
    for(int b = 0; b < bSize; b++) {
      sum += norm(link(row_idx, i, b));
    }
  }
  
#pragma unroll  
  for (int i = 0; i < ncol; i++) {
#pragma unroll
    for(int b = 0; b < bSize; b++) {
      link(row_idx, i, b) /= sqrt(sum);
    }
  }  
}

// orthogonalize two rows of the link:
template<int ncol>
static void orthogonalize(MDViewTp auto &link, const int row_idx_1, const int row_idx_2)
{
  constexpr int bSize = 1;//otherwise would need loop over bsize
  
  using data_tp = std::remove_cvref_t<decltype(link)>::value_type;
  
  std::complex<double> dot = 0.0;
  
#pragma once  
  for (int i = 0; i < ncol; i++) {
#pragma once
    for(int b = 0; b < bSize; b++) {
      dot += conj(link(row_idx_1, i, b)) * link(row_idx_2, i, b);
    }
  }
#pragma once  
  for (int i = 0; i < ncol; i++) {
#pragma once
    for(int b = 0; b < bSize; b++) {
      link(row_idx_2, i, b) -= static_cast<data_tp>(dot) * link(row_idx_1, i, b);
    }
  }
}

inline void accumulateConjugateProduct( ComplexTp auto &a, const ComplexTp auto &b, const ComplexTp auto &c, int sign) {
  using complex_tp = std::remove_cvref_t<decltype(a)>;  
  
  auto s = complex_tp{sign, -sign}; 
  //
  a += s*b*c;
}


// 0 -> STAGGERED_DSLASH,
// 1 -> ASQTAD_DSLASH
// 0 -> SU3_LINKS,
// 1 -> GENERAL_LINKS,
// 2 -> THREE_LINKS,
template <int dslash_type = 0, int link_type = 0> 
void constructRandomGaugeField(auto &gauge_field, const FloatTp auto extra_param){

  using data_tp = typename std::remove_cvref_t<decltype(gauge_field)>::data_tp;
  using real_tp = typename data_tp::value_type;  

  constexpr bool is_constant = false;
  constexpr int  bSize       = 1;
  constexpr int  nColor      = std::remove_cvref_t<decltype(gauge_field)>::Ncolor();
  //
  auto &&gauge_ref = gauge_field.View(); 

  using LinkAccessor = FieldAccessor<decltype(gauge_field.View()), is_constant, bSize>;//non-constant links   
  using Link         = LinkAccessor::LinkTp;  

  constexpr int nDir   = 4;
  
  const auto [Nx, Ny, Nz, Nt] = gauge_field.GetCBDims();  
  
  auto ids = impl::cartesian_4d_view({Nx, Ny, Nz, Nt}); 
  
  auto links = LinkAccessor(gauge_ref);  

  std::for_each(ids.begin(),
                ids.end(),
                [=] (const auto cartesian_idx) { 
                                      
                  auto X  = impl::convert_coords<4>(cartesian_idx);
                  //
#pragma unroll                  
                  for (int d = 0; d < nDir; d++) {             
                    //     
                    Link Ue = links(X, d, 0);
                    Link Uo = links(X, d, 1); 
                    //
                    auto Ue_view = Ue.view();
                    auto Uo_view = Uo.view();                    
                    
                    std::array<real_tp, 4> factors = {1.0, 1.0, 1.0, 1.0};
                    
                    if constexpr( link_type == 1 /*ASQTAD_LONG_LINKS*/) {
                      factors[1] = 2.0;
                      factors[2] = 3.0;
                      factors[3] = 4.0;
                    }
                    
#pragma unroll                    
                    for (int r = 0; r < nColor; r++ ) {
#pragma unroll                    
                      for (int c = 0; c < nColor; c++ ) {
#pragma unroll
                        for(int b = 0; b < bSize; b++) {                      
                          Ue_view(r,c,b) = data_tp(factors[0]*dis(gen), factors[1]*dis(gen));
                          Uo_view(r,c,b) = data_tp(factors[2]*dis(gen), factors[3]*dis(gen));                    
                        }
                      }
                    }//end for loops
                    //
                    if constexpr( link_type == 1 /*ASQTAD_LONG_LINKS*/) return;                    
                    //
                    normalize<nColor>(Ue_view, 1);
                    orthogonalize<nColor>(Ue_view, 1, 2);
                    normalize<nColor>(Ue_view, 2);
                    //                    
                    normalize<nColor>(Uo_view, 1);
                    orthogonalize<nColor>(Uo_view, 1, 2);
                    normalize<nColor>(Uo_view, 2);                                        
                    //
#pragma unroll                    
                    for (int c = 0; c < nColor; c++ ) {
                      for(int b = 0; b < bSize; b++) {                      
                        //
                        Ue_view(0,c,b) = data_tp(0.0);
                        Uo_view(0,c,b) = data_tp(0.0);                          
                      }
                    }//end for
#pragma unroll                    
                    for(int b = 0; b < bSize; b++) {                      
                      accumulateConjugateProduct(Ue_view(0, 0, b), Ue_view(1, 1, b), Ue_view(2, 2, b), +1);
                      accumulateConjugateProduct(Ue_view(0, 0, b), Ue_view(1, 2, b), Ue_view(2, 1, b), -1);                      
                      accumulateConjugateProduct(Ue_view(0, 1, b), Ue_view(1, 2, b), Ue_view(2, 0, b), +1);
                      accumulateConjugateProduct(Ue_view(0, 1, b), Ue_view(1, 0, b), Ue_view(2, 2, b), -1);                      
                      accumulateConjugateProduct(Ue_view(0, 2, b), Ue_view(1, 0, b), Ue_view(2, 1, b), +1);
                      accumulateConjugateProduct(Ue_view(0, 2, b), Ue_view(1, 1, b), Ue_view(2, 0, b), -1);                      
                      //
                      accumulateConjugateProduct(Uo_view(0, 0, b), Uo_view(1, 1, b), Uo_view(2, 2, b), +1);
                      accumulateConjugateProduct(Uo_view(0, 0, b), Uo_view(1, 2, b), Uo_view(2, 1, b), -1);                      
                      accumulateConjugateProduct(Uo_view(0, 1, b), Uo_view(1, 2, b), Uo_view(2, 0, b), +1);
                      accumulateConjugateProduct(Uo_view(0, 1, b), Uo_view(1, 0, b), Uo_view(2, 2, b), -1);                      
                      accumulateConjugateProduct(Uo_view(0, 2, b), Uo_view(1, 0, b), Uo_view(2, 1, b), +1);
                      accumulateConjugateProduct(Uo_view(0, 2, b), Uo_view(1, 1, b), Uo_view(2, 0, b), -1);                                           
                    }
                  }
                });  
  //              
  if constexpr ( link_type == 2/*ASQTAD_LONG_LINKS*/) {
    applyGaugeFieldScaling_long<dslash_type>(gauge_field, extra_param);
  } 
}

template <int dslash_type, int link_type, bool compute_fatlong = false>
void constructFatLongGaugeField(auto &fatlink, auto &longlink, const FloatTp auto anisotropy, const FloatTp auto tadpole_coeff, int test_type/*0,1,2*/) {  
  //
  constexpr bool is_constant = false;
  constexpr int  bSize       = 1;
  constexpr int  nDim        = 4;

  if (test_type == 0) {

    constructUnitGaugeField(fatlink,  anisotropy);
    constructUnitGaugeField(longlink, anisotropy);

    applyGaugeFieldScaling_long<dslash_type>(fatlink, tadpole_coeff);    

    if constexpr (dslash_type == 1 and  !compute_fatlong)
      applyGaugeFieldScaling_long<dslash_type>(longlink, tadpole_coeff);

  } else {
    // if doing naive staggered then set to long links so that the staggered phase is applied
    constexpr int fat_link_type = dslash_type == 1/*ASQTAD_DSLASH*/ ? 1 /*ASQTAD_FAT_LINKS*/ : 2 /* QUDA_ASQTAD_LONG_LINKS*/;

    constructRandomGaugeField<dslash_type, fat_link_type>(fatlink, tadpole_coeff);
      
    if constexpr (dslash_type == 1 /*QUDA_ASQTAD_DSLASH*/) {
      constexpr int long_link_type = 2/*ASQTAD_LONG_LINKS*/;      
      constructRandomGaugeField<dslash_type, long_link_type>(longlink, tadpole_coeff);
    }
    //
    if (dslash_type == 2/* QUDA_ASQTAD_DSLASH*/) {
      // incorporate non-trivial phase into long links
      const double phase      = (std::numbers::pi * dis(gen)) / std::numbers::pi ;
      const std::complex<double> z = std::polar(1.0, phase);
      //
      auto &&longlink_ref = longlink.View(); 

      using LinkAccessor = FieldAccessor<decltype(longlink.View()), is_constant, bSize>;//non-constant links   
      using Link         = LinkAccessor::LinkTp;  
  
      const auto [Nx, Ny, Nz, Nt] = longlink.GetCBDims();  
  
      auto ids = impl::cartesian_4d_view({Nx, Ny, Nz, Nt}); 
  
      auto llinks_accessor = LinkAccessor(longlink_ref);  

      std::for_each(ids.begin(),
                    ids.end(),
                    [=] (const auto cartesian_idx) {
                                      
                      auto X  = impl::convert_coords<4>(cartesian_idx);                    
                      //
#pragma unroll                  
                      for (int d = 0; d < nDim; d++) {             
                        //     
                        Link Ue = llinks_accessor(X, d, 0);
                        Link Uo = llinks_accessor(X, d, 1); 
                        //
                        Ue *= z;
                        Uo *= z;                                                                                                                                                
                      }
                    }); 
    }
  }
}


