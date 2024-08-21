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

template<std::size_t nDim, std::size_t nColor, std::size_t nParity = invalid_parity>
class GaugeFieldDescriptor {
  private:
    template <std::size_t src_nParity>
    static auto get_dims(const auto &src_dims){
      constexpr std::size_t dst_nParity = nParity; 
      
      std::array dim_{src_dims};
      
      if      constexpr (dst_nParity == 1 and src_nParity == 2) dim_[0] /= 2;
      else if constexpr (dst_nParity == 2 and src_nParity == 1) dim_[0] *= 2;      
      
      return dim_;
    }
    
    template <FieldType type, std::size_t nExtra, bool adjust_dim = true>
    static auto get_strides(const auto &d){
      const int d0 = adjust_dim ? (nParity == 2 ? d[0] / 2 : d[0]) : d[0];
      std::array<std::size_t, nDim+nExtra> strides{1, d0, d0*d[1], d0*d[1]*d[2], d0*d[1]*d[2]*d[3]};

      strides[nDim+1] =  strides[nDim+0]*nColor;                                                  
      strides[nDim+2] =  strides[nDim+1]*nColor;
      if constexpr (nParity == 2) { 
        strides[nDim+3] =  strides[nDim+2]*nDim;                             
      } 
      return strides;
    }    
    
  public:
    using ParityFieldDescriptor = GaugeFieldDescriptor<nDim, nColor, 1>;
 
    static constexpr std::size_t ndim   = nDim;                    // FIXME
    //
    static constexpr std::size_t ndir   = nDim;                    //must be 2*nDim (!)	  
    static constexpr std::size_t nspin  = invalid_spin;            //number of spin dof
    static constexpr std::size_t ncolor = nColor;                  //
    static constexpr std::size_t nparity= nParity;                 //    

    static constexpr FieldType  type = FieldType::VectorFieldType;

    //gauge  field extra dimensions: color + dims[ + parity, if par = 2 ] 
    //spinor field extra dimensions: color [ + parity, if par = 2 ]     
    static constexpr int nExtra      = (nparity == 2 ? 4 : 3);       

    const std::array<int, ndim> dim;    

    const FieldOrder         order  = FieldOrder::InvalidFieldOrder;        		
    const FieldParity        parity = FieldParity::InvalidFieldParity;//this is optional param
    const FieldBoundary      bc     = FieldBoundary::InvalidBC;

    std::shared_ptr<PMRBuffer> pmr_buffer;

    const std::array<std::size_t, ndim+nExtra> mdStrides;            //for mdspan views only

    GaugeFieldDescriptor()                             = default;
    GaugeFieldDescriptor(const GaugeFieldDescriptor& ) = default;
    GaugeFieldDescriptor(GaugeFieldDescriptor&& )      = default;

    GaugeFieldDescriptor(const std::array<int, ndim> dim, 
	                 const FieldParity     parity   = FieldParity::InvalidFieldParity,
	                 const FieldOrder      order    = FieldOrder::EOFieldOrder,
                         const FieldBoundary   bc       = FieldBoundary::InvalidBC,		    
	                 const bool is_exclusive        = true) : 
	                 dim{dim},
	                 order(order),
	                 parity(parity), 
		         bc(bc),
                         pmr_buffer(nullptr), 
                         mdStrides(get_strides<type,nExtra>(dir)) {
                         } 
                    
    template<typename Args>
    GaugeFieldDescriptor(const Args &args, const FieldParity parity) : 
                    dim(get_dims<std::remove_cvref_t<decltype(args)>::nparity>(args.dim)),
	            order(args.order),
	            parity(parity),
		    bc(args.bc),
                    pmr_buffer(args.pmr_buffer), 
                    mdStrides(get_strides<type,nExtra,false>(dir)) {
                    } 

    //Use it for block fields only:
    GaugeFieldDescriptor(const GaugeFieldDescriptor &args, 
                    const std::shared_ptr<PMRBuffer> extern_pmr_buffer) :
                    dim(args.dim),
                    order(args.order),
                    parity(args.parity),
		    bc(args.bc),
                    pmr_buffer(extern_pmr_buffer), 
                    mdStrides(args.mdStrides){ }
    
    decltype(auto) GetFieldSize() const {
      int vol = nDim*nColor*nColor; 
#pragma unroll      
      for(int i = 0; i < ndim; i++) vol *= dim[i];

      return vol;
    } 

    auto GetLatticeDims() const { return dim; }

    auto GetParityLatticeDims() const {
      std::array xcb{dim};
      xcb[0] = nParity == 2 ? xcb[0] / nParity : xcb[0];	    
      return xcb;
    }
    
    auto GetParity() const { return parity; }
    
    auto GetFieldSubset() const { return (nParity == 2 ? FieldSiteSubset::FullSiteSubset : (nParity == 1 ? FieldSiteSubset::ParitySiteSubset : FieldSiteSubset::InvalidSiteSubset)); }

    inline int  X(const int i) const { return dim[i]; }
    
    inline auto X() const { return dim; }    
    
 
    inline auto& GetMDStrides()  const { return mdStrides; }
  
    template<ArithmeticTp T, bool is_exclusive = true>
    void RegisterPMRBuffer(const bool is_reserved = false) {  
      // 
      const std::size_t nbytes = (GetFieldSize())*sizeof(T);
      //
      if (pmr_buffer != nullptr) pmr_buffer.reset(); 
      //
      pmr_buffer = pmr_pool::pmr_malloc(nbytes, is_reserved);
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

    auto operator=(const GaugeFieldDescriptor&) -> GaugeFieldDescriptor& = default;
    auto operator=(GaugeFieldDescriptor&&     ) -> GaugeFieldDescriptor& = default;
};

