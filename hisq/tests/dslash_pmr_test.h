#pragma once

using vector_tp         = std::vector<std::complex<Float>>;
using sloppy_vector_tp  = std::vector<std::complex<float>>;

using pmr_vector_tp         = impl::pmr::vector<std::complex<Float>>;
using sloppy_pmr_vector_tp  = impl::pmr::vector<std::complex<float>>;

void run_pmr_dslash_test(auto params, const auto dims, const int niter) {
  //
  constexpr int nSpinorParity = 2;
  constexpr int nGaugeParity  = 2;
  //
  constexpr bool clean_intermed_fields = true;
  //  
  const auto cs_param    = StaggeredSpinorFieldArgs<nSpinorParity>{dims, {0, 0, 0, 0}};
  //
  const auto gauge_param = GaugeFieldArgs<nGaugeParity>{dims, {0, 0, 0, 0}};

  // Create full precision gauge field:
  auto fat_lnks = create_field<vector_tp, decltype(gauge_param)>(gauge_param);
  //
  init_su3(fat_lnks);

  auto long_lnks = create_field<vector_tp, decltype(gauge_param)>(gauge_param);
  //
  init_su3(long_lnks);

  // Create low precision gauge field (NOTE: by setting copy_gauge = true we migrate data on the device):  
  constexpr bool copy_gauge = true;

  auto sloppy_fat_lnks  = create_field<decltype(fat_lnks), sloppy_vector_tp, copy_gauge>(fat_lnks);   

  auto sloppy_long_lnks = create_field<decltype(long_lnks), sloppy_vector_tp, copy_gauge>(long_lnks);  

  auto src_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);
  
  using cs_param_tp = decltype(src_spinor.ExportArg());
  
  auto dst_spinor  = create_field_with_buffer<sloppy_pmr_vector_tp, cs_param_tp>(cs_param);
  //
  init_spinor(src_spinor);  

  // Setup dslash arguments:
  auto &&fl_ref  = sloppy_fat_lnks.View();
  auto &&ll_ref  = sloppy_long_lnks.View();

  using sloppy_gauge_tp = decltype(sloppy_fat_lnks.View());

  std::unique_ptr<StaggeredDslashArgs<sloppy_gauge_tp>> hisq_args_ptr(new StaggeredDslashArgs{fl_ref, ll_ref});

  auto &hisq_args = *hisq_args_ptr;

  // Create dslash matrix
  auto mat = Mat<decltype(hisq_args), StaggeredDslash, decltype(params)>{hisq_args, params};

  using arg_tp = decltype(src_spinor.Even().ExportArg());  
  //
  constexpr bool do_warmup = false;
  //
  if constexpr (do_warmup) {
    mat(dst_spinor, src_spinor);
  }
  //
  auto wall_start = std::chrono::high_resolution_clock::now(); 
  
  for(int i = 0; i < niter; i++) {

    mat(dst_spinor, src_spinor);    
  }

  auto wall_stop = std::chrono::high_resolution_clock::now();

  auto wall_diff = wall_stop - wall_start;
  
  auto wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  constexpr bool do_check = false;  
  
  std::cout << "Done for EO version : time per iteration is > " << wall_time << "sec." << std::endl; 
  
  src_spinor.show();
  if constexpr (clean_intermed_fields) {
    dst_spinor.destroy();
    src_spinor.destroy();
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////////////////
  std::cout << "Run next src (re-use buffer)" << std::endl;

  auto src_spinor_v2   = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);
  auto dst_spinor_v2   = create_field_with_buffer<sloppy_pmr_vector_tp, decltype(cs_param)>(cs_param);

  wall_start = std::chrono::high_resolution_clock::now();   

  for(int i = 0; i < niter; i++) {
    // Apply dslash	  
    mat(dst_spinor_v2, src_spinor_v2);
  }
  
  wall_stop = std::chrono::high_resolution_clock::now();

  wall_diff = wall_stop - wall_start;
  
  wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for another EO version : time per iteration is > " << wall_time << "sec." << std::endl;  

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
void run_mrhs_pmr_dslash_test(auto params, const auto dims, const int niter) {
  // 
  constexpr int nSpinorParity = 2;
  constexpr int nGaugeParity  = 2;
  // 
  constexpr bool do_warmup = false; 
  //
  const auto cs_param = StaggeredSpinorFieldArgs<nSpinorParity>{dims, {0, 0, 0, 0}, FieldParity::InvalidFieldParity};
  //
  const auto gauge_param = GaugeFieldArgs<nGaugeParity>{dims, {0, 0, 0, 0}};
  //
  auto fat_lnks = create_field<vector_tp, decltype(gauge_param)>(gauge_param);
  //
  init_su3(fat_lnks);

  auto long_lnks = create_field<vector_tp, decltype(gauge_param)>(gauge_param);
  //
  init_su3(long_lnks);

  constexpr bool copy_gauge = true;
     
  auto sloppy_fat_lnks  = create_field<decltype(fat_lnks), sloppy_vector_tp, copy_gauge>(fat_lnks);   

  auto sloppy_long_lnks = create_field<decltype(long_lnks), sloppy_vector_tp, copy_gauge>(long_lnks);
  //
  // Setup dslash arguments:
  auto &&fl_ref  = sloppy_fat_lnks.View();
  auto &&ll_ref  = sloppy_long_lnks.View();

  using sloppy_gauge_tp = decltype(sloppy_fat_lnks.View());

  std::unique_ptr<StaggeredDslashArgs<sloppy_gauge_tp>> hisq_args_ptr(new StaggeredDslashArgs{fl_ref, ll_ref});

  auto &hisq_args = *hisq_args_ptr;

  // Create dslash matrix
  auto mat = Mat<decltype(hisq_args), StaggeredDslash, decltype(params)>{hisq_args, params};   
  //
  using sloppy_pmr_spinor_t  = Field<sloppy_pmr_vector_tp, decltype(cs_param)>;//

  auto src_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param) >(cs_param, N); 
  auto chk_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param) >(cs_param, N);

  for (int i = 0; i < src_block_spinor.nComponents(); i++) init_spinor( src_block_spinor.v[i] );
  
  auto dst_block_spinor = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param) >(cs_param, N);  
  //
  if constexpr (do_warmup) {
    mat(dst_block_spinor, src_block_spinor);    
  }

  auto wall_start = std::chrono::high_resolution_clock::now();   
 
  for(int i = 0; i < niter; i++) {
    mat(dst_block_spinor, src_block_spinor);      
  } 

  auto wall_stop = std::chrono::high_resolution_clock::now();

  auto wall_diff = wall_stop - wall_start;

  auto wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for MRHS version (N =  " << N << ") : time per iteration is > " << wall_time << "sec." << std::endl;

  
  src_block_spinor[0].show();
  //
  src_block_spinor.destroy();
  dst_block_spinor.destroy();  

  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////

  auto src_block_spinor_v2 = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param)>(cs_param, N);
  auto dst_block_spinor_v2 = create_block_spinor< sloppy_pmr_spinor_t, decltype(cs_param)>(cs_param, N);
  
  wall_start = std::chrono::high_resolution_clock::now();

  for(int i = 0; i < niter; i++) {
    mat(dst_block_spinor_v2, src_block_spinor_v2);
  }

  wall_stop = std::chrono::high_resolution_clock::now();

  wall_diff = wall_stop - wall_start;

  wall_time = (static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(wall_diff).count()) / 1e6)  / niter;

  std::cout << "Done for MRHS version (N =  " << N << ") : time per iteration is > " << wall_time << "sec." << std::endl;


  src_block_spinor_v2[0].show();

  src_block_spinor_v2.destroy();
  dst_block_spinor_v2.destroy();

  long_lnks.destroy();
  sloppy_long_lnks.destroy();
  
  fat_lnks.destroy();
  sloppy_fat_lnks.destroy(); 
}



