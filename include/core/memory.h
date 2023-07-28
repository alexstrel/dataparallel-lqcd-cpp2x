#pragma once

#include <vector>
#include <map>
#include <memory>
#include <memory_resource>
#include <ranges>

#include <core/common.h>

constexpr std::size_t alignment_req = std::alignment_of_v<long double>;

constexpr int max_n_buffers = 128;

template <typename T>  consteval  T zero() { return static_cast<T>{0 }; }
template <FloatTp  T>  consteval  T zero() { return static_cast<T>{0.f}; }
template <ComplexTp T> consteval  T zero() { return T{static_cast<T::value_type>(0.f), static_cast<T::value_type>(0.f)};  }

class UpstreamMemoryResource : public std::pmr::memory_resource {
  protected:
    //
    std::size_t nBytes = 0;
    
    void* do_allocate(std::size_t nbytes, std::size_t alignment = alignment_req) override {//alignof(std::max_align)
      nBytes = ( (nbytes + alignment - 1) / alignment ) *  alignment;
      //
      return std::aligned_alloc(alignment, nBytes);
    }
    
    void do_deallocate(void* ptr, std::size_t nbytes, std::size_t alignment = alignment_req) override {
      //
      return std::free(ptr);
    }
    
    bool do_is_equal(const std::pmr::memory_resource &other) const noexcept override { return (this == &other); } 
    
  public:
  
    UpstreamMemoryResource()                                = default;
    UpstreamMemoryResource(const UpstreamMemoryResource &)  = default;
    UpstreamMemoryResource(UpstreamMemoryResource &&)       = default;     
};

static auto upstream = UpstreamMemoryResource{};

class PMRBuffer{
  public:
    //
    std::size_t                          pmr_base_bytes;
    std::size_t                          pmr_bytes;
    std::shared_ptr<std::byte[]>         pmr_ptr;//needed only for pmr-used spinor    
    
    std::shared_ptr<std::pmr::monotonic_buffer_resource>  pmr_pool;    

    PMRState pmr_state = PMRState::InvalidState; 
    //
    PMRBuffer(const std::size_t nbytes, const PMRState state = PMRState::Vacant) : pmr_base_bytes(nbytes),
                                                                                   pmr_bytes(((nbytes + alignment_req - 1) / alignment_req ) *  alignment_req), 
                                                                                   pmr_ptr(static_cast<std::byte*>(std::aligned_alloc(alignment_req, pmr_bytes)), [](std::byte* p){std::free(p);}),
                                                                                   pmr_pool(std::make_shared<std::pmr::monotonic_buffer_resource>(pmr_ptr.get(), pmr_bytes)),                          
                                                                                   pmr_state(state) { } 

    PMRBuffer()                             = default;
    PMRBuffer(const PMRBuffer &)            = default;
    PMRBuffer(PMRBuffer &&)                 = default;

    virtual ~PMRBuffer()                    = default;

    auto Data()  const { return pmr_ptr;       }
    auto Get()   const { return pmr_ptr.get(); }
    
    auto BaseBytes()  const { return pmr_base_bytes; }        
    
    auto Bytes() const { return pmr_bytes; }    
    auto State() const { return pmr_state; }
    //
    auto Pool()  const { return pmr_pool;  } 
    
    bool IsExclusive() const { return (pmr_state == PMRState::LockedExclusive or pmr_state == PMRState::ReservedExclusive); }           
    
    void SetState(PMRState state) { pmr_state = state; }
    
    void ResetPool() const {  pmr_pool->release(); } 
    
    bool IsReserved(const std::size_t nbytes) const {  
      if (pmr_ptr != nullptr ) {
        if (pmr_state != PMRState::ReservedExclusive and pmr_state != PMRState::ReservedShared and pmr_state != PMRState::ReservedNonExclusive) {
          return false;
        }
        //
        return (nbytes <= pmr_base_bytes);
      } 
      //
      return false;
    } 
    
    // Simply return the state:
    bool IsReserved() const { return ((pmr_state == PMRState::ReservedExclusive) or (pmr_state == PMRState::ReservedShared) or (pmr_state == PMRState::ReservedNonExclusive));}
    
    void UpdateReservedState() {  if      (pmr_state == PMRState::ReservedExclusive)     pmr_state = PMRState::LockedExclusive;
                                  else if (pmr_state == PMRState::ReservedShared)        pmr_state = PMRState::Shared;
                                  else if (pmr_state == PMRState::ReservedNonExclusive)  pmr_state = PMRState::LockedNonExclusive;                                  
                                  else                                                   pmr_state = PMRState::InvalidState; }       
    
    void Release() {

      if(IsReserved()) { return; } //if reserved , just nop

      if      (pmr_state == PMRState::LockedNonExclusive)    pmr_state = PMRState::Shared;
      else                                                   pmr_state = PMRState::Vacant;//PMRState::LockedExclusive

      pmr_pool->release();
      //      
    }
               
    void Destroy() {  
      pmr_pool.reset();
      pmr_base_bytes = 0;
      pmr_bytes      = 0;      
      pmr_ptr.reset();
      //
      pmr_state = PMRState::InvalidState;
    } 
    
    bool IsEqual(const PMRBuffer &other) const { return (this->pmr_ptr.get() == other.Get()); }               
};

static PMRBuffer default_pmr_buffer = PMRBuffer();

enum class TargetMemorySpace { None = 0, Device = 1, Host = 2 };

namespace impl {
  namespace pmr {
  
  static TargetMemorySpace init_pmr_space = TargetMemorySpace::None;

  void SetDevicePMRSpace()  { init_pmr_space = TargetMemorySpace::Device; }
  void SetHostPMRSpace()    { init_pmr_space = TargetMemorySpace::Host;   }
  void SetDefaultPMRSpace() { init_pmr_space = TargetMemorySpace::None;   }  

  template <typename T>
  class vector {
    public:
      using allocator_type  = std::pmr::polymorphic_allocator<T>;
      //
      using value_type      = typename std::allocator_traits<allocator_type>::value_type;
      using size_type       = typename std::allocator_traits<allocator_type>::size_type;
      using difference_type = typename std::allocator_traits<allocator_type>::difference_type;
      //
      using reference       = value_type&;
      using const_reference = const value_type&;
      //
      using pointer         = typename std::allocator_traits<allocator_type>::pointer;
      using const_pointer   = typename std::allocator_traits<allocator_type>::const_pointer;
      //
      using iterator               = pointer;
      using const_iterator         = const_pointer;
      using reverse_iterator       = std::reverse_iterator<iterator>;
      using const_reverse_iterator = std::reverse_iterator<const_iterator>;
      
      vector() : data_(nullptr), size_(0), alloc_(std::pmr::get_default_resource()) {}
      

      explicit vector(size_type numElements, std::pmr::memory_resource* memResource) : data_(nullptr), 
                                                                                       size_(numElements), 
                                                                                       alloc_(memResource) {
                                                                                         data_ = alloc_.allocate(numElements);
											 //
											 if (init_pmr_space != TargetMemorySpace::None ) {
											   auto zero_ = zero<T>();
											   //
											   if (init_pmr_space == TargetMemorySpace::Device) {
											   //
											   	
											     std::fill(std::execution::par_unseq,
                                                                                                       this->begin(),
                                                                                                       this->end(),
                                                                                                       zero_);
  	  
											   } else if (init_pmr_space == TargetMemorySpace::Host) {
											   
                                                                                             std::fill(this->begin(),
                                                                                                       this->end(),
                                                                                                       zero_);
                                                                                                       
											   }
											 }   
                                                                                       }
      ~vector() {
        if(data_) {
          for(size_type i = 0; i < size_; ++i) {
            std::allocator_traits<allocator_type>::destroy(alloc_, data_ + i);
          }
          alloc_.deallocate(data_, size_);
        }
      }

      // Iterators
      constexpr iterator begin()                       noexcept { return data_; }
      constexpr const_iterator begin()           const noexcept { return data_; }
      constexpr const_iterator cbegin()          const noexcept { return data_; }
    
      constexpr iterator end()                         noexcept { return data_ + size_; }
      constexpr const_iterator end()             const noexcept { return data_ + size_; }
      constexpr const_iterator cend()            const noexcept { return data_ + size_; }

      constexpr reverse_iterator rbegin()              noexcept { return reverse_iterator(end()); }
      constexpr const_reverse_iterator rbegin()  const noexcept { return const_reverse_iterator(end()); }
      constexpr const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }

      constexpr reverse_iterator rend()                noexcept { return reverse_iterator(begin()); }
      constexpr const_reverse_iterator rend()    const noexcept { return const_reverse_iterator(begin()); }
      constexpr const_reverse_iterator crend()   const noexcept { return const_reverse_iterator(cbegin()); }

      constexpr pointer data() noexcept { return data_; }

      constexpr const_pointer data() const noexcept { return data_; }
      
      constexpr reference front() noexcept { return data_[0]; }      
      constexpr const_reference front() const noexcept { return data_[0]; }            

      constexpr reference back() noexcept { return data_[size_-1]; }      
      constexpr const_reference back() const noexcept { return data_[size_-1]; }            

      void resize(size_type newSize) {
        value_type* newData = alloc_.allocate(newSize);
        // Transfer data from old memory to new memory here...
        alloc_.deallocate(data_, size_);
        data_ = newData;
        size_ = newSize;
      }

      reference operator[](size_type index) {
        // we don't perform bounds check
        return data_[index];
      }

      const_reference operator[](size_type index) const { return data_[index]; }

      size_type size() const { return size_; }

    private:
      pointer        data_;
      size_type      size_;
      allocator_type alloc_;
  };
  

} // pmr
} // impl

/**
    Derived from the QUDA library, original developer : Kate Clark (NVIDIA)
*/

using PMRBufferRef = std::reference_wrapper<PMRBuffer>;

namespace pmr_pool
{
  /**
      Contains all instanses of pmr buffer (incl. unallocated)
  */
  static std::vector<std::shared_ptr<PMRBuffer>> pmrBuffers;
  
  /** Cache of pmr-memory allocations.  We cache pmr
      memory allocations so that fields can reuse them.*/
  static std::multimap<std::size_t, std::shared_ptr<PMRBuffer>> pmrCachedBuffers;

  /** Sizes of active pmr-memory allocations.  For convenience,
      we keep track of the sizes of active allocations (i.e., those not
      in the cache). */
  static std::map<std::byte*, std::shared_ptr<PMRBuffer>> pmrAllocatedBuffers;
  
  static bool pmr_memory_pool = false;   

  template<bool is_exclusive = true>
  decltype(auto) pmr_malloc(const std::size_t nbytes, const bool is_reserved = false) {
  
    PMRState final_buffer_state = is_reserved ? (is_exclusive ? PMRState::ReservedExclusive  : PMRState::ReservedShared) 
                                              : (is_exclusive ? PMRState::LockedExclusive    : PMRState::Shared        ); 
     
    if( pmr_memory_pool == false ) { 
      pmrBuffers.reserve(max_n_buffers);
      //
      pmr_memory_pool = true;
    }

    // state filter :
    auto select_pmr_buffer = [] (const auto &p) {
      auto &pmrb = p.second;
      return ( ((pmrb->State() == PMRState::Vacant) or (pmrb->State() == PMRState::Shared and not is_exclusive)) and (pmrb->State() != PMRState::ReservedExclusive and pmrb->State() != PMRState::ReservedShared and pmrb->State() != PMRState::ReservedNonExclusive) ); 
    };   

    // state transformation:
    auto transform_pmr_buffer_state = [=, &is_reserved, &final_buffer_state] (const auto &pmrb) mutable {
      if ( pmrb->State() == PMRState::Shared ) { /*this is a shared buffer*/
        pmrb->ResetPool();
        /* we locked buffer if it's not reserved: */
        if (not is_reserved) final_buffer_state = PMRState::LockedNonExclusive;
        else                 final_buffer_state = PMRState::ReservedNonExclusive;
      } /*otherwise an initial state was vacant, amd we can keep the origin values of final_buffer_state*/

      pmrb->SetState( final_buffer_state );
    };
   
    if ( not pmrCachedBuffers.empty() ) {//if buffer is not empty... 
      auto buffer_range = pmrCachedBuffers.equal_range(nbytes);      
          
      if ( buffer_range.first != pmrCachedBuffers.end() ) {//...and threre is one (not locked or reserved) with sufficient size
        //
        auto selected_pmr_buffers = std::ranges::subrange(buffer_range.first, buffer_range.second) | std::views::filter(select_pmr_buffer);

        if ( not selected_pmr_buffers.empty() ) {

          auto p = *std::ranges::begin(selected_pmr_buffers);
          auto& [bytes, pmr_buffer] = p;

          transform_pmr_buffer_state(pmr_buffer);

          return pmr_buffer;
        }
      }
    }
    //
    pmrBuffers.emplace_back(std::make_shared<PMRBuffer>(nbytes, final_buffer_state));
    //
    auto pmr_buffer = pmrBuffers.back();
    //
    pmrCachedBuffers.insert(std::make_pair(nbytes, pmr_buffer));	
    //
    pmrAllocatedBuffers.insert(std::make_pair(pmr_buffer->Get(), pmr_buffer));

    return pmr_buffer;
  }    

  template<ArithmeticTp T>
  void pmr_free(T* ptr) {
    //
    auto raw_ptr = reinterpret_cast<std::byte*>(ptr);
    
    if ( pmrAllocatedBuffers.empty() or (not pmrAllocatedBuffers.contains(raw_ptr)) ) { std::cerr << "Attempt to free invalid pmr buffer" << std::endl; }    
    
    auto it = pmrAllocatedBuffers.find(raw_ptr);
    
    if (it != pmrAllocatedBuffers.end()) {
    
      auto& pmr_buffer   = it->second; 
      //
      if (pmr_buffer->State() != PMRState::Vacant) { std::cerr << "Attempt to free pmr buffer that is still in use." << std::endl; }           
      //
      std::size_t nbytes = pmr_buffer->BaseBytes();
      //
      auto range = pmrCachedBuffers.equal_range(nbytes);      
      //
      for (auto& range_it = range.first; range_it != range.second; ++range_it) {
        auto &cached_pmr_buffer = range_it->second;
        if ( cached_pmr_buffer->IsEquel(pmr_buffer) ) {
          //
          pmrCachedBuffers.erase(range_it);                    
          //
          break;
        }
      }     
      //
      pmr_buffer->Destroy();
      //
      pmrAllocatedBuffers.erase(it);          
    }    
  }

  void flush_pmr() {
  
    if (pmr_memory_pool) {
    
      pmrAllocatedBuffers.clear();
      pmrCachedBuffers.clear();      
      //
      for(auto &pmr_buffer : pmrBuffers) { 
        pmr_buffer->Destroy();  
        //        
        pmr_buffer.reset();
      }
      //
      pmrBuffers.resize(0);
      
      pmr_memory_pool = false;
    }
  }
} // namespace pool


