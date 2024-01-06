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
  } else if constexpr (nD == invalid_dir and nS == invalid_spin  and nC != invalid_color) {
    return FieldType::StaggeredSpinorFieldType;	  
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
      
      if      constexpr (dst_nParity == 1 and src_nParity == 2) dir_[0] /= 2;
      else if constexpr (dst_nParity == 2 and src_nParity == 1) dir_[0] *= 2;      
      
      return dir_;
    }
    
    template <FieldType type, std::size_t nExtra, bool adjust_dir = true>
    static auto get_strides(const auto &d){
      const int d0 = adjust_dir ? (nParity == 2 ? d[0] / 2 : d[0]) : d[0];
      std::array<std::size_t, nDim+nExtra> strides{1, d0, d0*d[1], d0*d[1]*d[2], d0*d[1]*d[2]*d[3]};

      if constexpr (nParity == 2) {
        if constexpr (type == FieldType::VectorFieldType) {
          strides[nDim+1] =  strides[nDim+0]*nColor;                                                  
          strides[nDim+2] =  strides[nDim+1]*nColor; 
          strides[nDim+3] =  strides[nDim+2]*nDir;                             
        } else if constexpr (type == FieldType::StaggeredSpinorFieldType){ //spinor
          strides[nDim+1] =  strides[nDim+0]*nColor;                                                                            
        }
      } else if constexpr (nParity == 1) {
        if constexpr (type == FieldType::VectorFieldType) {
          strides[nDim+1] =  strides[nDim+0]*nColor;                                                  
          strides[nDim+2] =  strides[nDim+1]*nColor;                                                                              
        }
      } 
      return strides;
    }    
    
  public: 
    static constexpr std::size_t ndim   = nDim;                    // FIXME
    //
    static constexpr std::size_t ndir   = nDir;                    //vector field dim   (2 for U1 gauge)	  
    static constexpr std::size_t nspin  = nSpin;                   //number of spin dof (2 for spinor)
    static constexpr std::size_t ncolor = nColor;                  //for all fields
    static constexpr std::size_t nparity= nParity;                 //for all fields    

    static constexpr FieldType  type = get_field_type<ndir, nspin, ncolor>();

    //gauge  field extra dimensions: color + dirs[ + parity, if par = 2 ] 
    //spinor field extra dimensions: color [ + parity, if par = 2 ]     
    static constexpr int nExtra      = type == FieldType::VectorFieldType ? (nparity == 2 ? 4 : 3) : (nparity == 2 ? 2 : 1);       

    const std::array<int, ndim> dir;    

    const FieldOrder         order  = FieldOrder::InvalidFieldOrder;        		
    const FieldParity        parity = FieldParity::InvalidFieldParity;//this is optional param
    const FieldBoundary      bc     = FieldBoundary::InvalidBC;

    std::shared_ptr<PMRBuffer> pmr_buffer;

    const std::array<std::size_t, ndim+nExtra> mdStrides;            //for mdspan views only

    FieldDescriptor()                        = default;
    FieldDescriptor(const FieldDescriptor& ) = default;
    FieldDescriptor(FieldDescriptor&& )      = default;

    FieldDescriptor(const std::array<int, ndim> dir, 
	            const FieldParity     parity   = FieldParity::InvalidFieldParity,
	            const FieldOrder      order    = FieldOrder::EOFieldOrder,
                    const FieldBoundary   bc       = FieldBoundary::InvalidBC,		    
	            const bool is_exclusive        = true) : 
	            dir{dir},
	            order(order),
	            parity(parity), 
		    bc(bc),
                    pmr_buffer(nullptr), 
                    mdStrides(get_strides<type,nExtra>(dir)) {
                    } 
                    
    template<typename Args>
    FieldDescriptor(const Args &args, const FieldParity parity) : 
                    dir(get_dims<std::remove_cvref_t<decltype(args)>::nparity>(args.dir)),
	            order(args.order),
	            parity(parity),
		    bc(args.bc),
                    pmr_buffer(args.pmr_buffer), 
                    mdStrides(get_strides<type,nExtra,false>(dir)) {
                    } 

    //Use it for block fields only:
    FieldDescriptor(const FieldDescriptor &args, 
                    const std::shared_ptr<PMRBuffer> extern_pmr_buffer) :
                    dir(args.dir),
                    order(args.order),
                    parity(args.parity),
		    bc(args.bc),
                    pmr_buffer(extern_pmr_buffer), 
                    mdStrides(args.mdStrides){ }
    
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
      } else if constexpr (type == FieldType::StaggeredSpinorFieldType) {
	return vol*nColor;	
      }
      //
      return static_cast<std::size_t>(0);
    } 

    auto GetLatticeDims() const { return dir; }

    auto GetParityLatticeDims() const {
      std::array xcb{dir};
      xcb[0] = nParity == 2 ? xcb[0] / nParity : xcb[0];	    
      return xcb;
    }
    
    auto GetParity() const { return parity; }
    
    auto GetFieldSubset() const { return (nParity == 2 ? FieldSiteSubset::FullSiteSubset : (nParity == 1 ? FieldSiteSubset::ParitySiteSubset : FieldSiteSubset::InvalidSiteSubset)); }

    inline int  X(const int i) const { return dir[i]; }
    
    inline auto X() const { return dir; }    
    
 
    inline auto& GetMDStrides()      const { return mdStrides; }
  
    template<ArithmeticTp T, bool is_exclusive = true>
    void RegisterPMRBuffer(const bool is_reserved = false) {  
      // 
      const std::size_t nbytes = (GetFieldSize())*sizeof(T);
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
      const std::size_t nbytes = (GetFieldSize())*sizeof(T);    
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

constexpr std::size_t ndims   = 4;
constexpr std::size_t ncolors = 3;
constexpr std::size_t ndirs   = 4;

template<int nParity = invalid_parity> using GaugeFieldArgs           = FieldDescriptor<ndims, ndirs, invalid_spin, ncolors, nParity>;

template<int nParity = invalid_parity> using StaggeredSpinorFieldArgs = FieldDescriptor<ndims, invalid_dir, invalid_spin, ncolors, nParity>;

