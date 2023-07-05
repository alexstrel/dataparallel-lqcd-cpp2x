#pragma once

#include <assert.h>
#include <algorithm>
#include <ranges>
#include <execution>
#include <numeric>
#include <memory>
#include <memory_resource>
#include <functional>

#include <fields/field_concepts.h>
#include <core/enums.h>
#include <core/memory.h>


template<std::size_t nD, std::size_t nS, std::size_t nC>
consteval FieldType get_field_type() {

  if constexpr (nD != invalid_dir and nS == invalid_spin  and nC != invalid_color){
    return FieldType::VectorFieldType;	  
  } else if constexpr (nD == invalid_dir and nS != invalid_spin  and nC != invalid_color) {
    return FieldType::SpinorFieldType;	  
  }

  return FieldType::InvalidFieldType;
}

template<std::size_t nDim, std::size_t nDir = invalid_dir, std::size_t nSpin = invalid_spin, std::size_t nColor = invalid_color, std::size_t nParity = invalid_parity>
class FieldDescriptor {
  private:
    template <std::size_t src_nParity>
    static auto get_dims(const auto &src_dir){
      constexpr std::size_t dst_nParity = nParity; 
      
      std::array dir_{src_dir};
      
      if constexpr (dst_nParity == 1 and src_nParity == 2) dir_[0] /= 2;
      
      return dir_;
    }
    
  public: 
    static constexpr std::size_t ndim   = nDim;                       // FIXME
    //
    static constexpr std::size_t ndir   = nDir;                    //vector field dim   (2 for U1 gauge)	  
    static constexpr std::size_t nspin  = nSpin;                   //number of spin dof (2 for spinor)
    static constexpr std::size_t ncolor = nColor;                  //for all fields
    static constexpr std::size_t nparity= nParity;                       //for all fields    

    static constexpr FieldType  type = get_field_type<ndir, nspin, ncolor>();

    static constexpr int nFace       = type == FieldType::VectorFieldType ? 1 : 2; 

    const std::array<int, ndim> dir;
    const std::array<int, nFace*ndim> comm_dir;

    const FieldOrder         order  = FieldOrder::InvalidFieldOrder;        		
    const FieldParity        parity = FieldParity::InvalidFieldParity;//this is optional param

    std::shared_ptr<PMRBuffer> pmr_buffer;

    FieldDescriptor()                        = default;
    FieldDescriptor(const FieldDescriptor& ) = default;
    FieldDescriptor(FieldDescriptor&& )      = default;

    FieldDescriptor(const std::array<int, ndim> dir, 
                    const std::array<int, ndim*nFace> comm_dir,  
	            const FieldParity     parity   = FieldParity::InvalidFieldParity,
	            const FieldOrder      order    = FieldOrder::EOFieldOrder,	            
	            const bool is_exclusive        = true) : 
	            dir{dir},
                    comm_dir{comm_dir},
	            order(order),
	            parity(parity), 
                    pmr_buffer(nullptr){ 
                      if (parity != FieldParity::InvalidFieldParity and nparity != 1) {
                        std::cerr << "Incorrect number of parities " << std::endl;
                        std::quick_exit( EXIT_FAILURE );
                      }
                    } 
                    
    template<typename Args>
    FieldDescriptor(const Args &args, const FieldParity parity) : 
                    dir(get_dims<std::remove_cvref_t<decltype(args)>::nparity>(args.dir)),
	            comm_dir([&dir_=args.dir,dst_nParity=nparity, src_nParity=std::remove_cvref_t<decltype(args)>::nparity]()->std::array<int, ndim*nFace> {
                        std::array<int, nFace*ndim> comm_dir{};

                        for( int d = 0; d < ndim; d++){
                          const bool do_div = (d != 0) and (dst_nParity == 1 and src_nParity == 2);
                          for( int face = 0; face < nFace; face++ ) {
                             comm_dir[d+face*nFace] =  do_div ? dir_[d+face*nFace] / 2 : 1;
                          }
                        } return comm_dir;
                      }()),
	            order(args.order),
	            parity(parity),
                    pmr_buffer(args.pmr_buffer){ 
                      if (parity != FieldParity::InvalidFieldParity and nparity != 1) {
                        std::cerr << "Incorrect number of parities " << std::endl;
                        std::quick_exit( EXIT_FAILURE );
                      }                    
                    } 

    //Use it for block fields only:
    FieldDescriptor(const FieldDescriptor &args, 
                    const std::shared_ptr<PMRBuffer> extern_pmr_buffer) :
                    dir(args.dir),
                    comm_dir(args.comm_dir),
                    order(args.order),
                    parity(args.parity),
                    pmr_buffer(extern_pmr_buffer){ }
       

    decltype(auto) GetFieldSize() const {
      int vol = 1; 
#pragma unroll      
      for(int i = 0; i < ndim; i++) vol *= dir[i];
      
      if  constexpr (type == FieldType::ScalarFieldType) {
        return vol;
      } else if constexpr (type == FieldType::VectorFieldType) {
	return vol*nDir*nColor*nColor;
      } else if constexpr (type == FieldType::SpinorFieldType) {
	return vol*nSpin*nColor;
      }
      //
      return static_cast<std::size_t>(0);
    } 

    decltype(auto) GetGhostZoneSize() const {
      int vol = 1; 
#pragma unroll      
      for(int i = 0; i < ndim; i++) vol *= dir[i];
      
      if  constexpr (type == FieldType::ScalarFieldType) {
        return vol*nFace;
      } else if constexpr (type == FieldType::VectorFieldType) {
        return vol*nDir*nColor*nColor*nFace;
      } else if constexpr (type == FieldType::SpinorFieldType) {
        return vol*nSpin*nColor*nFace;
      }
      return static_cast<std::size_t>(0);
    }

    auto GetFaceSize(int i, int face_idx = 0) const {
      if  constexpr (type == FieldType::ScalarFieldType) {
        return comm_dir[i*nFace+face_idx];
      } else if constexpr (type == FieldType::VectorFieldType) {
	return comm_dir[i*nFace+face_idx]*nColor*nColor;
      } else if constexpr (type == FieldType::SpinorFieldType) {
	return comm_dir[i*nFace+face_idx]*nSpin*nColor;
      }
      //
      return static_cast<std::size_t>(0);
    }

    auto GetLatticeDims() const {
      return dir;	    
    }

    auto GetParityLatticeDims() const {
      std::array xcb{dir};
      xcb[0] = nParity == 2 ? xcb[0] / nParity : xcb[0];	    
      return xcb;
    }
    
    auto GetParity() const { return parity; }
    
    auto GetFieldSubset() const { return (nParity == 2 ? FieldSiteSubset::FullSiteSubset : (nParity == 1 ? FieldSiteSubset::ParitySiteSubset : FieldSiteSubset::InvalidSiteSubset)); }

    inline int  X(const int i) const { return dir[i]; }
    
    inline auto X() const { return dir; }    
    
    template<ArithmeticTp T, bool is_exclusive = true>
    void RegisterPMRBuffer(const bool is_reserved = false) {  
      // 
      const std::size_t nbytes = (GetFieldSize()+GetGhostZoneSize())*sizeof(T);
      //
      if (pmr_buffer != nullptr) pmr_buffer.reset(); 
      //
      pmr_buffer = pmr_pool::pmr_malloc<is_exclusive>(nbytes, is_reserved);
    }    

    void UnregisterPMRBuffer() {
      //
      if(pmr_buffer != nullptr) { 
        pmr_buffer->Release();
        //
        pmr_buffer.reset(); 
      }
    }
    
    void ResetPMRBuffer() {
      //
      if(pmr_buffer != nullptr) { 
        pmr_buffer.reset(); 
      }
    }

    void ReleasePMRBuffer() const { if(pmr_buffer != nullptr) pmr_buffer->Release(); }     
    
    template<ArithmeticTp T>    
    bool IsReservedPMR() const {
      //
      const std::size_t nbytes = (GetFieldSize()+GetGhostZoneSize())*sizeof(T);    
      //
      return pmr_buffer->IsReserved(nbytes);
    } 

    bool IsExclusive() const { 
      if (pmr_buffer != nullptr) return pmr_buffer->IsExclusive(); 
      //
      return false;
    }

    auto GetState() const { 
      if (pmr_buffer != nullptr) {
        return pmr_buffer->State(); 
      }

      return PMRState::InvalidState;
    } 

    auto SetState(PMRState state) { 
      if (pmr_buffer != nullptr) { pmr_buffer->SetState(state); }
    }
    
    void UpdatedReservedPMR() const { pmr_buffer->UpdateReservedState(); }

    auto operator=(const FieldDescriptor&) -> FieldDescriptor& = default;
    auto operator=(FieldDescriptor&&     ) -> FieldDescriptor& = default;
};

constexpr std::size_t ndims   = 2;
constexpr std::size_t nspins  = 2;
constexpr std::size_t ncolors = 1;
constexpr std::size_t ndirs   = 2;

template<int nParity = invalid_parity> using GaugeFieldArgs  = FieldDescriptor<ndims, ndirs, invalid_spin, ncolors, nParity>;

template<int nParity = invalid_parity> using SpinorFieldArgs = FieldDescriptor<ndims, invalid_dir, nspins, ncolors, nParity>;

