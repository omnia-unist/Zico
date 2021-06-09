/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_BFC_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_BFC_ALLOCATOR_H_

#include <pthread.h>
#include <stdio.h>
#include <sys/timeb.h>
#include <unistd.h>

#include <array>
#include <cstdlib>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/common_runtime/gangmuk_parser.h"
#include "tensorflow/core/common_runtime/global_pool.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

// A memory allocator that implements a 'best-fit with coalescing'
// algorithm.  This is essentially a very simple version of Doug Lea's
// malloc (dlmalloc).
//
// The goal of this allocator is to support defragmentation via
// coalescing.  One assumption we make is that the process using this
// allocator owns pretty much all of the memory, and that nearly
// all requests to allocate memory go through this interface.
class BFCAllocator : public Allocator {
 public:
  // Takes ownership of sub_allocator.
  BFCAllocator(SubAllocator* sub_allocator, size_t total_memory,
               bool allow_growth, const string& name);

  // Gangmuk: gpu_option 추가로 받음
  BFCAllocator(SubAllocator* sub_allocator, size_t total_memory,
               bool allow_growth, const string& name,
               const GPUOptions& gpu_options);

  ~BFCAllocator() override;

  // Gangmuk
  void PrintOverhead(string caller, size_t size, uint64 elapsed);

  string Name() override { return name_; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;

  // Gangmuk: Newly Added 3

  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr) override;

  // Gangmuk: Newly Added 3
  void* AllocateRaw(size_t alignment, size_t num_bytes, 
                    std::map< std::string, std::string > node_info) override;

  void* AllocateRaw(size_t alignment, size_t num_bytes,
                    const AllocationAttributes& allocation_attr,
                    std::map< std::string, std::string > node_info) override;

  void DeallocateRaw(void* ptr) override;

  bool TracksAllocationSizes() override;

  size_t RequestedSize(const void* ptr) override;

  size_t AllocatedSize(const void* ptr) override;

  int64 AllocationId(const void* ptr) override;

  void GetStats(AllocatorStats* stats) override;

  void ClearStats() override;

  void RecordNodeInfoToChunk(void* ptr,
                             std::map<std::string, std::string> node_info);

  unsigned long long iter_start_time;

  std::string SharableToString(bool sharable) {
    if (sharable) {
      return "sharable";
    }
    return "non_sharable";
  }

  bool GetSharableFromNodeInfo(std::map<std::string, std::string> node_info) {
    if (share_feature_map_only) {
      /* output 텐서이고, 커널이름이 gradient가 "아니면" sharable */
      if (node_info["caller"] == "allocate_output" &&
          node_info["kernel_name"].find("gradients") == std::string::npos) {
        return true;
      }
      return false;
    } else {
      /* output 텐서면 sharable */
      if (node_info["caller"] == "allocate_output") {
        return true;
      }
      return false;
    }
  }

  bool GetIsGradientFromNodeInfo(std::map<std::string, std::string> node_info) {
    if (node_info["kernel_name"].find("gradients") != std::string::npos) {
      return true;
    }
    return false;
  }

  // Gangmuk
  const GPUOptions gpu_options;

  int bfc_id = -1;

  int bookkeeper_id = -1;
  bool print_summary = true;

  /*
    1: custom millisecond clock (baseTime)
    2: TF's microsecond clock (base_micro)
  */
  int which_clock = 2;

	/*
	  get millisecond time: to See Allocation Time and Deallocation Time
	*/
	int getMilliCount();
	int getMilliSpan(int nTimeStart);
	int getTimeDiff();
	int baseTime = getMilliCount();

  static unsigned long long base_micro;

	bool first_time_flag;
  int unique_id = 0;

  /*
    shared GlobalPool object between All BFCAllocator
  */
  static cr_GlobalPool cr_global_pool_;

  /*
    adaptive_alloc divides training into two phase based on step_id.
    From stabilized_step, BFC enteres into phase two.
    in-place update profiling phase is activated on step 10.
    That's why BFC phase two is triggered from step 11.
  */
  bool adaptive_alloc;
  int train_steps = 0;
  // int bfc_phase_two_step = 20;
  int print_summary_step;
  int allocation_info_registration_step;
  int print_bfc_overhead = 0;
  int print_bfc_overhead_step = 100;
  int start_aligning_step;

  /*
    IF 'activate_global_pool' is true, then you must turn on one of global_pool_release flag.
    Only one out of two global_pool_release can be true.
  */
  bool activate_global_pool;
  // bool global_pool_release_ar;
  // bool global_pool_release_all;
  bool share_feature_map_only = false;

  // Deprecated //
  /*
    Only one out of two gpu_release can be true.
  */
  // bool gpu_release_ar;  // It releases free "Activation" AllocationRegion only.
  // bool gpu_release_all; // It releases all the free AlloationRegion, regardless of act-AR or nonact-AR.
  
  /*
    AllocationRegion size related flag 
  */
  bool update_curr_region_allocation_size = true; // Decrease curr_AR size when releasing AR.
  bool tighten_ar; // To make AllocationRegions size fit to chunk size 'everytime'.
  size_t default_ar_size; // 256MB. only with tighten_ar_size = true.
  // size_t default_ar_size = 1 << 20; // 1MB. only with tighten_ar_size = true.

  /*
    signal variables for second phase entering.
    It becomes globally phase two, only when all bfc allocators become phase two.
    (second phase == phase two)
  */
  static int num_gpu_bfc;
  bool local_phase_two;
  static bool bfc_0_phase_two_signal;
  static bool bfc_1_phase_two_signal;
  static bool activate_phase_two;

  // current step id comes from op_kernel.cc packed within node_info.
  int current_step_id = 0;
  
  /*
    mutex for Global Pool
  */
  bool lock_global_pool = true; //
  static std::mutex global_pool_mtx;
  static std::mutex sub_allocator_mtx;

  int b_to_mb = 1048576;

  bool debug_print = false;

 private:
  
  struct Bin;

  void* AllocateRawInternal(size_t alignment, size_t num_bytes,
                            bool dump_log_on_failure);

  // Gangmuk: Newly Added 3
  void* AllocateRawInternal(size_t alignment, size_t num_bytes,
                            bool dump_log_on_failure,
                            std::map< std::string, std::string > node_info);

  void DeallocateRawInternal(void* ptr);

  // A ChunkHandle is an index into the chunks_ vector in BFCAllocator
  // kInvalidChunkHandle means an invalid chunk
  typedef size_t ChunkHandle;
  static const int kInvalidChunkHandle = -1;

  typedef int BinNum;
  static const int kInvalidBinNum = -1;
  static const int kNumBins = 21;

  // A Chunk points to a piece of memory that's either entirely free or entirely
  // in use by one user memory allocation.
  //
  // An AllocationRegion's memory is split up into one or more disjoint Chunks,
  // which together cover the whole region without gaps.  Chunks participate in
  // a doubly-linked list, and the prev/next pointers point to the physically
  // adjacent chunks.
  //
  // Since a chunk cannot be partially in use, we may need to split a free chunk
  // in order to service a user allocation.  We always merge adjacent free
  // chunks.
  //
  // Chunks contain information about whether they are in use or whether they
  // are free, and contain a pointer to the bin they are in.
  struct Chunk {
    // Gangmuk
		//  0: initial value
		//  1: bridge
		//  2: non-bridge
		//  3: others
    int is_ACT = 0;
    size_t size = 0;  // Full size of buffer.

    std::string caller;
    std::string kernel_name;
    bool is_gradient;
    bool sharable;
    unsigned long long birth_time;
    unsigned long long death_time;

    // We sometimes give chunks that are larger than needed to reduce
    // fragmentation.  requested_size keeps track of what the client
    // actually wanted so we can understand whether our splitting
    // strategy is efficient.
    size_t requested_size = 0;

    // allocation_id is set to -1 when the chunk is not in use. It is assigned a
    // value greater than zero before the chunk is returned from
    // AllocateRaw, and this value is unique among values assigned by
    // the parent allocator.
    int64 allocation_id = -1;
    void* ptr = nullptr;  // pointer to granted subbuffer.

    // If not kInvalidChunkHandle, the memory referred to by 'prev' is directly
    // preceding the memory used by this chunk.  E.g., It should start
    // at 'ptr - prev->size'
    ChunkHandle prev = kInvalidChunkHandle;

    // If not kInvalidChunkHandle, the memory referred to by 'next' is directly
    // following the memory used by this chunk.  E.g., It should be at
    // 'ptr + size'
    ChunkHandle next = kInvalidChunkHandle;

    // What bin are we in?
    BinNum bin_num = kInvalidBinNum;

    bool in_use() const { return allocation_id != -1; }

    string DebugString(BFCAllocator* a,
                       bool recurse) NO_THREAD_SAFETY_ANALYSIS {
      string dbg;
      strings::StrAppend(
          &dbg, "  Size: ", strings::HumanReadableNumBytes(size),
          " | Requested Size: ", strings::HumanReadableNumBytes(requested_size),
          " | in_use: ", in_use());
      if (recurse && prev != BFCAllocator::kInvalidChunkHandle) {
        Chunk* p = a->ChunkFromHandle(prev);
        strings::StrAppend(&dbg, ", prev: ", p->DebugString(a, false));
      }
      if (recurse && next != BFCAllocator::kInvalidChunkHandle) {
        Chunk* n = a->ChunkFromHandle(next);
        strings::StrAppend(&dbg, ", next: ", n->DebugString(a, false));
      }
      return dbg;
    }
  };

  // A Bin is a collection of similar-sized free chunks.
  struct Bin {
    // All chunks in this bin have >= bin_size memory.
    size_t bin_size = 0;

    struct ChunkComparator {
      explicit ChunkComparator(BFCAllocator* allocator)
          : allocator_(allocator) {}
      // Sort first by size and then use pointer address as a tie breaker.
      bool operator()(const ChunkHandle ha,
                      const ChunkHandle hb) const NO_THREAD_SAFETY_ANALYSIS {
        const Chunk* a = allocator_->ChunkFromHandle(ha);
        const Chunk* b = allocator_->ChunkFromHandle(hb);
        if (a->size != b->size) {
          return a->size < b->size;
        }
        return a->ptr < b->ptr;
      }

     private:
      BFCAllocator* allocator_;  // The parent allocator
    };

    typedef std::set<ChunkHandle, ChunkComparator> FreeChunkSet;
    // List of free chunks within the bin, sorted by chunk size.
    // Chunk * not owned.
    FreeChunkSet free_chunks;
    Bin(BFCAllocator* allocator, size_t bs)
        : bin_size(bs), free_chunks(ChunkComparator(allocator)) {}
  };

  static const size_t kMinAllocationBits = 8;
  static const size_t kMinAllocationSize = 1 << kMinAllocationBits;

  // BFCAllocator allocates memory into a collection of disjoint
  // AllocationRegions.  Each AllocationRegion corresponds to one call to
  // SubAllocator::Alloc().
  //
  // An AllocationRegion contains one or more Chunks, covering all of its
  // memory.  Its primary job is to map a pointers to ChunkHandles.
  //
  // This class is thread-compatible.
  class AllocationRegion {
   public:
    // AllocationRegion(void* ptr, size_t memory_size, size_t region_id)
    //       : ptr_(ptr),
    //         memory_size_(memory_size),
    //         region_id_(region_id),
    //         end_ptr_(
    //             static_cast<void*>(static_cast<char*>(ptr_) + memory_size_)) {
    //     DCHECK_EQ(0, memory_size % kMinAllocationSize);
    //     const size_t n_handles =
    //         (memory_size + kMinAllocationSize - 1) / kMinAllocationSize;
    //     handles_.reset(new ChunkHandle[n_handles]);
    //     for (size_t i = 0; i < n_handles; i++) {
    //       handles_[i] = kInvalidChunkHandle;
    //     }
    //     if (print_ar) {
    //       printf("AllocationRegion-original(size: %zuMB, region_id: %zu)\n",
    //               ptr, memory_size / (1024 * 1024), region_id);
    //     }
    //   }

    // Gangmuk: ctor with memset
    AllocationRegion(void* ptr, size_t memory_size, size_t region_id, bool sharable_)
        : ptr_(ptr),
          memory_size_(memory_size),
          region_id_(region_id),
          end_ptr_(static_cast<void*>(static_cast<char*>(ptr_) + memory_size_)),
          sharable(sharable_)
    {
      DCHECK_EQ(0, memory_size % kMinAllocationSize);
      const size_t n_handles =
          (memory_size + kMinAllocationSize - 1) / kMinAllocationSize;
      handles_.reset(new ChunkHandle[n_handles]);
      memset(handles_.get(), kInvalidChunkHandle, n_handles);
    }

    bool print_ar = false;

    AllocationRegion() = default;
    AllocationRegion(AllocationRegion&& other) { Swap(other); }
    AllocationRegion& operator=(AllocationRegion&& other) {
      Swap(other);
      return *this;
    }

    void* ptr() const { return ptr_; }
    void* end_ptr() const { return end_ptr_; }
    size_t memory_size() const { return memory_size_; }
    size_t region_id() const { return region_id_; }
    int random_id() const { return random_id_; }

    ChunkHandle get_handle(const void* p) const {
      return handles_[IndexFor(p)];
    }

    int size_to_MB() { return memory_size_ / (1024 * 1024); }

    void set_handle(const void* p, ChunkHandle h) {
      if (print_ar) {
        printf("$$ AR[%zu] set_handle()\n", region_id_);
      }
      handles_[IndexFor(p)] = h;
    }

    bool erase(const void* p) { set_handle(p, kInvalidChunkHandle); }

    // Gangmuk: Originally, it was private variable.
    std::unique_ptr<ChunkHandle[]> handles_;  // handles_ of AllocationRegion class

    // Gangmuk
    mutable std::string AR_type = "Uninitialized";
    bool sharable;

   private:
    void Swap(AllocationRegion& other) {
      std::swap(ptr_, other.ptr_);
      std::swap(memory_size_, other.memory_size_);
      std::swap(end_ptr_, other.end_ptr_);
      std::swap(handles_, other.handles_);
      std::swap(region_id_, other.region_id_);
    }

    int IndexFor(const void* p) const {
      std::uintptr_t p_int = reinterpret_cast<std::uintptr_t>(p);
      std::uintptr_t base_int = reinterpret_cast<std::uintptr_t>(ptr_);
      DCHECK_GE(p_int, base_int);
      DCHECK_LT(p_int, base_int + memory_size_);
      return static_cast<int>(((p_int - base_int) >> kMinAllocationBits));
    }

    // Metadata about the allocation region.
    void* ptr_ = nullptr;
    size_t memory_size_ = 0;
    void* end_ptr_ = nullptr;
    size_t region_id_ = 0;
    int random_id_ = std::rand() % 10000;

    /* gangmuk newly added variables */
    // ACT
    unsigned int allocated_size_ = 0;
    bool first_flag = true;
    int id_;

    // Array of size "memory_size / kMinAllocationSize".  It is
    // indexed by (p-base) / kMinAllocationSize, contains ChunkHandle
    // for the memory allocation represented by "p"

    //////////////////////////////////////////////
    TF_DISALLOW_COPY_AND_ASSIGN(AllocationRegion);
    //////////////////////////////////////////////
  };
  // end of public for class AllocationRegion

  // Gangmuk
  // CAUTION: Declaring private again.
  private: 
  // RegionManager aggregates one or more "AllocationRegions" and provides
  // a layer of indirection from pointers to the underlying ChunkHandle,
  // allowing allocation across multiple discontiguous memory regions.
  //
  // This class is thread-compatible.
  class RegionManager {
   public:
    RegionManager() {
		}
    ~RegionManager() {}

    bool print_rm = false;
    int region_id=1;

    // Gangmuk: ACT,Overloading 2, region_id_from_gp,
    void AddAllocationRegion(void* memory_ptr, size_t memory_size, 
                             size_t region_id_from_gp, bool sharable_, 
                             bool from_gp) {
      // Insert sorted by end_ptr
      auto entry = std::upper_bound(regions_.begin(), regions_.end(), memory_ptr, &Comparator);
			size_t region_id_;

			// Activation AR from GlobalPool
			if (sharable_ && from_gp == true) {
				region_id_ = region_id_from_gp;
			}
			// Activation AR from GPU
			else {
				region_id_ = region_id;
				region_id++;
			}
      regions_.insert(entry, AllocationRegion(memory_ptr, memory_size, region_id_, sharable_));
    }

    /*
      완전히 내가 새로 넣은 함수. 원래는 AR을 BFC에서 프리하지 않으니 필요없는
      함수다. 그런데 dynamic하게 BFC로부터 AR을 Free하고 싶기 때문에,regions_
      에서 Free된 AR을 적절하게 삭제 시켜야됨. AddAllocationRegion()함수를
      따라서 만든 거임.

      그런데 지금 이게 제대로 작동하는지 확신이 없다.
      확인필요.
    */
    void EraseAllocationRegion(void* memory_ptr, size_t region_id, bool ACT_AR) {
      // printf("EraseAllocationRegion before, %zu, %d\n", region_id, ACT_AR);
      // fflush(stdout);
      auto entry = std::upper_bound(regions_.begin(), regions_.end(), memory_ptr, &Comparator);
      regions_.erase(entry);
      // printf("EraseAllocationRegion after, %zu, %d\n", region_id, ACT_AR);
      // fflush(stdout);
    }

    ChunkHandle get_handle(const void* p) const {
      return RegionFor(p)->get_handle(p);
    }

  
    void set_handle(const void* p, ChunkHandle h) {
      return MutableRegionFor(p)->set_handle(p, h);
    }

    // It calls AllocationRegion.erase(p)
    int erase(const void* p) { return MutableRegionFor(p)->erase(p); }

    const std::vector<AllocationRegion>& regions() const { return regions_; }

    /* gangmuk */
    // moved from private to public field to call from bfc_allocator.cc's DeleteChunk function
    const AllocationRegion* RegionFor(const void* p) const {
      auto entry =
          std::upper_bound(regions_.begin(), regions_.end(), p, &Comparator);

      if (entry != regions_.end()) {
        return &(*entry);
      }

      LOG(FATAL) << "Could not find Region for " << p;
      return nullptr;
    }

    const size_t get_AR_id(const void* p) const{
      const AllocationRegion* AR_ = RegionFor(p);
      return AR_->region_id();
    }


   private:
    static bool Comparator(const void* ptr, const AllocationRegion& other) {
      return ptr < other.end_ptr();
    }

    AllocationRegion* MutableRegionFor(const void* p) {
      return const_cast<AllocationRegion*>(RegionFor(p));
    }

//    const AllocationRegion* RegionFor(const void* p) const {
//      auto entry =
//          std::upper_bound(regions_.begin(), regions_.end(), p, &Comparator);
//
//      if (entry != regions_.end()) {
//        return &(*entry);
//      }
//
//      LOG(FATAL) << "Could not find Region for " << p;
//      return nullptr;
//    }

   private:
    std::vector<AllocationRegion> regions_;

  }; // end of class RegionManager

  // Returns 'bytes' rounded up to the next highest kMinAllocationSize.
  static size_t RoundedBytes(size_t bytes);

  // Gangmuk
  void RegisterNewAllocationRegion(
      size_t bytes, void* mem_addr, size_t region_id,
      std::map<std::string, std::string> node_info, bool from_gp);

  // Try to add a new memory region that can satisfy an allocation of
  // 'rounded_bytes' bytes.  Returns true on success and false on
  // failure.
  bool Extend(size_t alignment, size_t rounded_bytes)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Gangmuk: ACT
  bool Extend(size_t alignment, size_t rounded_bytes,
              std::map<std::string, std::string> node_info)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Returns a pointer to an underlying allocated chunk of size
  // 'rounded_bytes'.
  void* FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Gangmuk
  void* FindChunkPtr(BinNum bin_num, size_t rounded_bytes, size_t num_bytes,
                     std::map<std::string, std::string> node_info)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Splits the chunk specified by 'h' into two chunks, one at least
  // of size 'num_bytes'.
  void SplitChunk(ChunkHandle h, size_t num_bytes)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Merges the two chunk handles.  Requires that the chunks are
  // contiguous in their allocation.
  int Merge(ChunkHandle h, ChunkHandle h2) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Frees the memory represented by 'h', coalescing the chunk if
  // possible.
  void FreeAndMaybeCoalesce(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

//  // Adds the chunk 'h' to the proper free bin.
//  void InsertFreeChunkIntoBin(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Adds the chunk 'h' to the proper free bin.
  void InsertFreeChunkIntoBin(ChunkHandle h, bool sharable) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Removes the free chunk pointed to by 'c' from the set free_chunks.
  void RemoveFreeChunkIterFromBin(Bin::FreeChunkSet* free_chunks,
                                  const Bin::FreeChunkSet::iterator& c)
      EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Removes a free chunk from the bin.
  void RemoveFreeChunkFromBin(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Removes the chunk metadata represented by 'h'.
  // int DeleteChunk(ChunkHandle h, ChunkHandle h1) EXCLUSIVE_LOCKS_REQUIRED(lock_);
  int DeleteChunk(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Try to free empty AllocationRegion
  int TryToFreeAllocationRegion(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  string RenderOccupancy() EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void DumpMemoryLog(size_t num_bytes, bool print_only) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  ChunkHandle AllocateChunk() EXCLUSIVE_LOCKS_REQUIRED(lock_);
  void DeallocateChunk(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  Chunk* ChunkFromHandle(ChunkHandle h) EXCLUSIVE_LOCKS_REQUIRED(lock_);

  // Information about a Bin that is useful for debugging.
  struct BinDebugInfo {
    size_t total_bytes_in_use = 0;
    size_t total_bytes_in_bin = 0;
    size_t total_requested_bytes_in_use = 0;
    size_t total_chunks_in_use = 0;
    size_t total_chunks_in_bin = 0;
  };

  // Computes and returns a BinDebugInfo for each Bin.
  std::array<BinDebugInfo, kNumBins> get_bin_debug_info()
      EXCLUSIVE_LOCKS_REQUIRED(lock_);

  AllocatorRetry retry_helper_;

  // Structures immutable after construction
  size_t memory_limit_ = 0;

  inline int Log2FloorNonZeroSlow(uint64 n) {
    int r = 0;
    while (n > 0) {
      r++;
      n >>= 1;
    }
    return r - 1;
  }

  // Returns floor(log2(n)).
  inline int Log2FloorNonZero(uint64 n) {
#if defined(__GNUC__)
    return 63 ^ __builtin_clzll(n);
#elif defined(PLATFORM_WINDOWS) && (_WIN64)
    unsigned long index;
    _BitScanReverse64(&index, n);
    return index;
#else
    return Log2FloorNonZeroSlow(n);
#endif
  }

  // Map from bin size to Bin
  Bin* BinFromIndex(BinNum index) {
    return reinterpret_cast<Bin*>(&(bins_space_[index * sizeof(Bin)]));
  }
	
	// Gangmuk: Bin separation
	// overloading BinFromIndex
  Bin* BinFromIndex(BinNum index, bool sharable) {
    if (sharable) {
      return reinterpret_cast<Bin*>(&(act_bins_space_[index * sizeof(Bin)]));
      // return reinterpret_cast<Bin*>(&(bins_space_[index * sizeof(Bin)]));
    } else {
      return reinterpret_cast<Bin*>(&(bins_space_[index * sizeof(Bin)]));
    }
  }

  size_t BinNumToSize(BinNum index) {
    return static_cast<size_t>(256) << index;
  }

  BinNum BinNumForSize(size_t bytes) {
    uint64 v = std::max<size_t>(bytes, 256) >> kMinAllocationBits;
    int b = std::min(kNumBins - 1, Log2FloorNonZero(v));
    return b;
  }

  Bin* BinForSize(size_t bytes) { return BinFromIndex(BinNumForSize(bytes)); }

  Bin* BinForSize(size_t bytes, bool sharable) {
    return BinFromIndex(BinNumForSize(bytes), sharable);
  }

  char bins_space_[sizeof(Bin) * kNumBins];

	// Gangmuk: Bin separation
//  char nonact_bins_space_[sizeof(Bin) * kNumBins];
  char act_bins_space_[sizeof(Bin) * kNumBins];

  // The size of the current region allocation.
  size_t curr_region_allocation_bytes_;
  // Gangmuk
  size_t curr_region_allocation_bytes_ACT_;

  // The total number of allocated bytes by the allocator.
  size_t total_region_allocated_bytes_ = 0;
  size_t total_sharable_region_allocated_bytes_ = 0;
  size_t total_nonsharable_region_allocated_bytes_ = 0;

  // An indicator that expansion of a region has hit the limits
  // of the available memory.
  bool started_backpedal_ = false;

  std::unique_ptr<SubAllocator> sub_allocator_;
  string name_;

  // Structures mutable after construction
  mutable mutex lock_;
  RegionManager region_manager_ GUARDED_BY(lock_);

  std::vector<Chunk> chunks_ GUARDED_BY(lock_);

  // Pointer to head of linked list of free Chunks
  ChunkHandle free_chunks_list_ GUARDED_BY(lock_);

  // Counter containing the next unique identifier to assign to a
  // newly-created chunk.
  int64 next_allocation_id_ GUARDED_BY(lock_);

  // Stats.
  AllocatorStats stats_ GUARDED_BY(lock_);

  friend class GPUBFCAllocatorPrivateMethodsTest;
  TF_DISALLOW_COPY_AND_ASSIGN(BFCAllocator);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_BFC_ALLOCATOR_H_