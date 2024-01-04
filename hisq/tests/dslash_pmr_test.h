#pragma once

using vector_tp         = std::vector<std::complex<Float>>;
using sloppy_vector_tp  = std::vector<std::complex<float>>;

using pmr_vector_tp         = impl::pmr::vector<std::complex<Float>>;
using sloppy_pmr_vector_tp  = impl::pmr::vector<std::complex<float>>;

#include <staggered_dslash_reference.h> 
#include <improved_staggered_dslash_reference.h>

constexpr int nColors  = 3;
constexpr int mv_flops = (8 * nColors - 2) * nColors;
constexpr int num_dir  = 2 * 4;

void run_pmr_dslash_test(auto params, const auto dims, const int niter, const int test_type) {
  //
  const int vol = dims[0]*dims[1]*dims[2]*dims[3] / 2;
  //
  auto gflop = (( (2*num_dir*mv_flops + (2*num_dir-1)*2*3 /*accumulation*/ + 2*2*3 /*xpay flops*/)*vol)) * 1e-9 ;
  //
  constexpr FieldParity parity   = FieldParity::EvenFieldParity;
  //
  constexpr int nSrcSpinorParity = 2;
  //
  constexpr int nGaugeParity     = 2;
  //
  constexpr bool clean_intermed_fields = true;
  //  
  const auto src_cs_param = StaggeredSpinorFieldArgs<nSrcSpinorParity>{dims};
  //
  const auto gauge_param  = GaugeFieldArgs<nGaugeParity>{dims};

  // Create full precision gauge field:
  GaugeField auto fat_lnks  = create_field<vector_tp, decltype(gauge_param)>(gauge_param);
  //
  GaugeField auto long_lnks = create_field<vector_tp, decltype(gauge_param)>(gauge_param);
  //
#if 0
  constructFatLongGaugeField<1, 2>(fat_lnks, long_lnks, 0.5, 5.0, test_type);
#else
  init_su3(fat_lnks);
  init_su3(long_lnks);
#endif

  // Create low precision gauge field (NOTE: by setting copy_gauge = true we migrate data on the device):  
  constexpr bool copy_gauge = true;

  GaugeField auto sloppy_fat_lnks  = create_field<decltype(fat_lnks), sloppy_vector_tp, copy_gauge>(fat_lnks);   

  GaugeField auto sloppy_long_lnks = create_field<decltype(long_lnks), sloppy_vector_tp, copy_gauge>(long_lnks);  
  //
  FullSpinorField auto src_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(src_cs_param)>(src_cs_param);
  //
  const auto dst_cs_param = src_spinor.ExportParityArg(parity);
  //
  ParitySpinorField auto dst_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(dst_cs_param)>(dst_cs_param);
  //
  ParitySpinorField auto chk_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(dst_cs_param)>(dst_cs_param);
  //
  init_spinor(src_spinor);  

  // Setup dslash arguments:
  GaugeField auto &&fl_ref  = sloppy_fat_lnks.View();
  GaugeField auto &&ll_ref  = sloppy_long_lnks.View();

  constexpr bool do_arg_conversion = true;
  constexpr bool is_improved       = false;

  using sloppy_gauge_tp = decltype(sloppy_fat_lnks.View());

  using StaggeredArgs = StaggeredDslashArgs<sloppy_gauge_tp, do_arg_conversion, is_improved>;

  std::unique_ptr<StaggeredArgs> hisq_args_ptr(new StaggeredArgs{fl_ref, ll_ref});

  auto &hisq_args = *hisq_args_ptr;

  // Create dslash matrix
  auto mat = Mat<decltype(hisq_args), StaggeredDslash, decltype(params)>{hisq_args, params, parity};  
  //
  const bool do_warmup = true; 
  //
  if (do_warmup) {
    mat(dst_spinor, src_spinor.Even());
  }
  std::cout << "Begin bench \n" << std::endl;
  //
  ParitySpinorField auto &&dst_view = dst_spinor.View();
  //
  const ParitySpinorField auto &&src_view = src_spinor.Even();
 
  auto wall_start = std::chrono::high_resolution_clock::now(); 
  
  for(int i = 0; i < niter; i++) mat(dst_view, src_view); 

  auto wall_stop = std::chrono::high_resolution_clock::now();

  auto wall_diff = wall_stop - wall_start;
  
  auto wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  constexpr bool do_check = false;

  if constexpr (do_check) { 
    const int parity_bit = parity == FieldParity::EvenFieldParity ? 0 : 1;
#if 1  
    StaggeredDslashRef<float>(chk_spinor.View(), src_spinor.Even(),  src_spinor.Even(), sloppy_fat_lnks, params.M, chk_spinor.GetCBDims(), parity_bit); 
#else
    ImprovedDslashRef<float>(chk_spinor, src_spinor.Even(),  src_spinor.Even(), sloppy_fat_lnks, sloppy_long_lnks, params.M, chk_spinor.GetCBDims(), parity_bit);
#endif    


    auto &&chk = chk_spinor.Accessor();
    auto &&dst = dst_spinor.Accessor();     
    //
    check_field(chk, dst, 5e-6);
  }    
  
  std::cout << "Done for EO version : time per iteration is > " << wall_time << "sec." << std::endl; 
  std::cout << "Flops > " << gflop / wall_time << " gflops." << std::endl;

  src_spinor.show();
  if constexpr (clean_intermed_fields) {
    dst_spinor.destroy();
    src_spinor.destroy();
    chk_spinor.destroy();    
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "Run next src (re-use buffer)" << std::endl;

  auto src_spinor_v2   = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(src_cs_param)>(src_cs_param);
  auto dst_spinor_v2   = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(dst_cs_param)>(dst_cs_param);

  wall_start = std::chrono::high_resolution_clock::now();   

  for(int i = 0; i < niter; i++) {
    // Apply dslash	  
    mat(dst_spinor_v2.View(), src_spinor_v2.Even());
  }
  
  wall_stop = std::chrono::high_resolution_clock::now();

  wall_diff = wall_stop - wall_start;
  
  wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for another EO version : time per iteration is > " << wall_time << "sec." << std::endl;  

  std::cout << "Flops > " << gflop / wall_time << " gflops." << std::endl;

  long_lnks.destroy();
  sloppy_long_lnks.destroy();
  
  fat_lnks.destroy();
  sloppy_fat_lnks.destroy();  
  
  src_spinor_v2.show();  
  
  dst_spinor_v2.destroy();
  src_spinor_v2.destroy();  

  if constexpr (not clean_intermed_fields) {
    dst_spinor.destroy();
    src_spinor.destroy();
  }
}

template<int N>
void run_mrhs_pmr_dslash_test(auto params, const auto dims, const int niter, const int test_type) {
  //
  const int vol = dims[0]*dims[1]*dims[2]*dims[3] / 2;
  //
  auto gflop = (( (2*num_dir*mv_flops + (2*num_dir-1)*2*3 /*accumulation*/ + 2*2*3 /*xpay flops*/)*vol)) * 1e-9;//gflops per component
  //
  constexpr int nSpinorParity = 1;
  constexpr int nGaugeParity  = 2;
  // 
  constexpr bool do_warmup = true; 
  //
  const auto cs_param = StaggeredSpinorFieldArgs<nSpinorParity>{dims,FieldParity::EvenFieldParity};
  //
  const auto gauge_param = GaugeFieldArgs<nGaugeParity>{dims};
  //
  GaugeField auto fat_lnks = create_field<vector_tp, decltype(gauge_param)>(gauge_param);

  GaugeField auto long_lnks = create_field<vector_tp, decltype(gauge_param)>(gauge_param);
#if 0
  constructFatLongGaugeField<1, 2>(fat_lnks, long_lnks, 0.5, 5.0, test_type);
#else
  init_su3(fat_lnks);
  init_su3(long_lnks);
#endif

  constexpr bool copy_gauge = true;
     
  GaugeField auto sloppy_fat_lnks  = create_field<decltype(fat_lnks), sloppy_vector_tp, copy_gauge>(fat_lnks);   

  GaugeField auto sloppy_long_lnks = create_field<decltype(long_lnks), sloppy_vector_tp, copy_gauge>(long_lnks);
  //
  // Setup dslash arguments:
  GaugeField auto &&fl_ref  = sloppy_fat_lnks.View();
  GaugeField auto &&ll_ref  = sloppy_long_lnks.View();

  using sloppy_gauge_tp = decltype(sloppy_fat_lnks.View());

  std::unique_ptr<StaggeredDslashArgs<sloppy_gauge_tp>> hisq_args_ptr(new StaggeredDslashArgs{fl_ref, ll_ref});

  auto &hisq_args = *hisq_args_ptr;

  // Create dslash matrix
  auto mat = Mat<decltype(hisq_args), StaggeredDslash, decltype(params)>{hisq_args, params};   
  //
  using sloppy_pmr_spinor_t  = Field<sloppy_pmr_vector_tp, decltype(cs_param)>;//

  BlockParitySpinorField auto src_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param) >(cs_param, N); 
  BlockParitySpinorField auto chk_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param) >(cs_param, N);

  for (int i = 0; i < src_block_spinor.nComponents(); i++) init_spinor( src_block_spinor.v[i] );
  
  BlockParitySpinorField auto dst_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param) >(cs_param, N);  
  //
  if constexpr (do_warmup) {
    mat(dst_block_spinor, src_block_spinor);    
  }

  std::cout << "Begin bench \n" << std::endl;

  auto wall_start = std::chrono::high_resolution_clock::now();   
 
  for(int i = 0; i < niter; i++) {
    mat(dst_block_spinor, src_block_spinor);      
  } 

  auto wall_stop = std::chrono::high_resolution_clock::now();

  auto wall_diff = wall_stop - wall_start;

  auto wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for MRHS version (N =  " << N << ") : time per iteration is > " << wall_time << "sec." << std::endl;

  std::cout << "Flops per component > " << gflop / (wall_time / N) << " gflops." << std::endl;
  
  src_block_spinor[0].show();
  //
  src_block_spinor.destroy();
  dst_block_spinor.destroy();  

  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////

  BlockParitySpinorField auto src_block_spinor_v2 = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param)>(cs_param, N);
  BlockParitySpinorField auto dst_block_spinor_v2 = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param)>(cs_param, N);
  
  wall_start = std::chrono::high_resolution_clock::now();

  for(int i = 0; i < niter; i++) {
    mat(dst_block_spinor_v2, src_block_spinor_v2);
  }

  wall_stop = std::chrono::high_resolution_clock::now();

  wall_diff = wall_stop - wall_start;

  wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for MRHS version (N =  " << N << ") : time per iteration is > " << wall_time << "sec." << std::endl;

  std::cout << "Flops per component > " << gflop / (wall_time / N) << " gflops." << std::endl;

  src_block_spinor_v2[0].show();

  src_block_spinor_v2.destroy();
  dst_block_spinor_v2.destroy();

  long_lnks.destroy();
  sloppy_long_lnks.destroy();
  
  fat_lnks.destroy();
  sloppy_fat_lnks.destroy(); 
}


