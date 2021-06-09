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

#include <atomic>

#include "tensorflow/core/common_runtime/bfc_allocator.h"

#include "tensorflow/core/common_runtime/allocator_retry.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include <ctime>
#include <chrono>
#include <sys/timeb.h>
#include <unistd.h>

#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)
#endif

namespace tensorflow {

// Gangmuk
// Declaring static variable in cpp file.
// If you miss it, compilation will fail.
unsigned long long BFCAllocator::base_micro = Env::Default()->NowMicros();
cr_GlobalPool BFCAllocator::cr_global_pool_;
std::mutex BFCAllocator::global_pool_mtx;
std::mutex BFCAllocator::sub_allocator_mtx;
size_t BFCAllocator::global_max_chunk_size;

// BFCAllocator for CUDA_host will use following constructor
BFCAllocator::BFCAllocator(SubAllocator* sub_allocator, size_t total_memory,
                           bool allow_growth, const string& name)
    : sub_allocator_(sub_allocator),
      name_(name),
      free_chunks_list_(kInvalidChunkHandle),
      next_allocation_id_(1),
      gangmuk_next_allocation_id_(1) {

  printf("*********** %s-BFCAllocator() without GlobalPool !\n", name.c_str());
  adaptive_alloc = false;
  activate_global_pool = false;
  release_all = false;
  tensor_classification = false;
  // one_bin = parser.parse_target_config("OneBin");
  one_bin = true;
  // global_pool_release_ar = false;
  // global_pool_release_all = false;
  // gpu_release_ar = false;
  // gpu_release_all = false;
  tighten_ar = true;
  default_ar_size = 128;  // meaningless because tightnn_ar is false
  // default_ar_size = parser.parse_target_config("DefaultRegionSize");
  default_ar_size = default_ar_size << 20;
  // bfc_phase_two_step = 99999;  // meaningless because adaptive_alloc is false
  print_bfc = 0;
  print_bfc_step = 0;
  print_bfc_overhead_step = 0;
  one_allocator = false;
  // coordinator_policy = "not_init";

  phase_two_ = false;

  // added by gangmuk to set allow_growth true as default option
  allow_growth = true;
  if (allow_growth) {
    // 1MiB smallest initial allocation, unless total memory available
    // is less.
    curr_region_allocation_bytes_ =
        RoundedBytes(std::min(total_memory, size_t{1048576}));
  } else {
    curr_region_allocation_bytes_ = RoundedBytes(total_memory);
    printf("@@ WRONG PATH: you have to set allow_growth flag\n");
    exit(-1);
  }

  // Allocate the requested amount of memory.
  memory_limit_ = total_memory;
  printf("%s,memory_limit_,%zu,%zu\n", name_.c_str(), memory_limit_, memory_limit_/(1024*1024));
  stats_.bytes_limit = static_cast<int64>(total_memory);

  // Create a bunch of bins of various good sizes.

  /**** Bin Separation ****/

  // I(Gangmuk) create "two" bins to fit all possible ranges that cover the
  // memory_limit_ starting from allocations up to 256 bytes to
  // allocations up to (and including) the memory limit.

  /* Bin for activation chunk */
  for (BinNum b = 0; b < kNumBins; b++) {
    size_t bin_size = BinNumToSize(b);
    VLOG(1) << "Creating bin of max chunk size " << strings::HumanReadableNumBytes(bin_size);

    bool sharable_ = true;
    new (BinFromIndex(b, sharable_)) Bin(this, bin_size);
    CHECK_EQ(BinForSize(bin_size, sharable_), BinFromIndex(b, sharable_));
    CHECK_EQ(BinForSize(bin_size + 255, sharable_), BinFromIndex(b, sharable_));
    CHECK_EQ(BinForSize(bin_size * 2 - 1, sharable_), BinFromIndex(b, sharable_));
    if (b + 1 < kNumBins) {
      CHECK_NE(BinForSize(bin_size * 2, sharable_), BinFromIndex(b, sharable_));
    }
  }

	/* Bin for non-activation chunk */
  for (BinNum b = 0; b < kNumBins; b++) {
    size_t bin_size = BinNumToSize(b);
    VLOG(1) << "Creating bin of max chunk size "
            << strings::HumanReadableNumBytes(bin_size);

    bool sharable_ = false;
    new (BinFromIndex(b, sharable_)) Bin(this, bin_size);
    CHECK_EQ(BinForSize(bin_size, sharable_), BinFromIndex(b, sharable_));
    CHECK_EQ(BinForSize(bin_size + 255, sharable_), BinFromIndex(b, sharable_));
    CHECK_EQ(BinForSize(bin_size * 2 - 1, sharable_), BinFromIndex(b, sharable_));
    if (b + 1 < kNumBins) {
      CHECK_NE(BinForSize(bin_size * 2, sharable_), BinFromIndex(b, sharable_));
    }
  }
}

// Gangmuk: gpu_options 추가로 받음.
// BFCAllocator for GPU will use following constructor
BFCAllocator::BFCAllocator(SubAllocator* sub_allocator, size_t total_memory,
                           bool allow_growth, const string& name,
                           const GPUOptions& gpu_options)
    : sub_allocator_(sub_allocator),
      name_(name),
      free_chunks_list_(kInvalidChunkHandle),
      next_allocation_id_(1),
      gangmuk_next_allocation_id_(1) {
  printf("*********** %s-BFCAllocator() ************ \n", name.c_str());
  adaptive_alloc = parser.parse_target_config("AdaptiveAlloc");
  activate_global_pool = parser.parse_target_config("ActivateGlobalPool");
  release_all = parser.parse_target_config("ReleaseAll");
  tensor_classification = parser.parse_target_config("TensorClassification");
  one_bin = parser.parse_target_config("OneBin");
  // global_pool_release_ar = parser.parse_target_config("GlobalPoolReleaseAR");
  // global_pool_release_all = parser.parse_target_config("GlobalPoolReleaseAll");
  // share_feature_map_only = parser.parse_target_config("ShareFeatureMapOnly");
  share_feature_map_only = false;
  // gpu_release_ar = parser.parse_target_config("GpuReleaseAR");
  // gpu_release_all = parser.parse_target_config("GpuReleaseAll");
  tighten_ar = parser.parse_target_config("TightenRegion");
  // default_ar_size = parser.parse_target_config("DefaultRegionSize");
  float default_region_size_float = parser.parse_float_config("DefaultRegionSize");
  size_t default_region_size = default_region_size_float;
  if (default_region_size_float < 0) {
    // size_t one_mb = 1 << 20;
    // default_ar_size = one_mb * default_region_size_float;
    default_ar_size = 1 << 18; // 256KB
  }
  else if (default_region_size_float > 0){
    default_ar_size = default_region_size << 20;
  }
  else {
    default_ar_size = 1 << 19; // 512KB
  }
  printf("DefaultRegionSizeMB,%zu\n", default_ar_size/(1024*1024));
  
  size_t mini_default_ar_size_bytes = 256;
  mini_default_ar_size = mini_default_ar_size_bytes << 20;

  // bfc_phase_two_step = parser.parse_target_config("BFCPhaseTwoStep");
  print_bfc = parser.parse_target_config("PrintBFC");
  print_bfc_step = parser.parse_target_config("PrintBFCStep");
  print_bfc_overhead_step = parser.parse_target_config("PrintBFCOverheadStep");
  cuda_free_start_step = 0;
  start_aligning_step = parser.parse_target_config("StartAligningStep");
  assert(start_aligning_step > cuda_free_start_step);
  one_allocator = parser.parse_target_config("OneAllocator");
  // coordinator_policy = parser.parse_string_config("CoordinatorPolicy");
  phase_two_ = false;

  if (Name() == "GPU_0_bfc") {
    bfc_id = 0;
  } else if (Name() == "GPU_1_bfc") {
    bfc_id = 1;
  } else if (Name() == "GPU_2_bfc") {
    bfc_id = 2;
  } else if (Name() == "GPU_3_bfc") {
    bfc_id = 3;
  }


  // Gangmuk: Force allow_growth
  allow_growth = true;
  if (allow_growth) {
    // 1MiB smallest initial allocation, unless total memory available
    // is less.
    curr_region_allocation_bytes_ =
        RoundedBytes(std::min(total_memory, size_t{1048576}));
  } else {
    curr_region_allocation_bytes_ = RoundedBytes(total_memory);
    printf("@@ WRONG PATH: you have to set allow_growth flag\n");
    exit(-1);
  }

  // Allocate the requested amount of memory.
  memory_limit_ = total_memory;
  printf("%s,memory_limit_,%zu,%zu\n", name_.c_str(), memory_limit_,
         memory_limit_ / (1024 * 1024));
  stats_.bytes_limit = static_cast<int64>(total_memory);

  // Gangmuk
  cr_global_pool_.InitGlobalPool(base_micro, memory_limit_);
  cr_global_pool_.AddNewBFCStatus(bfc_id, Name());

  // Create a bunch of bins of various good sizes.

  /**** Bin Separation ****/
  // Gangmuk: It creates "two" bins to fit all possible ranges that cover the
  // memory_limit_ starting from allocations up to 256 bytes to
  // allocations up to (and including) the memory limit.

  /* Bin for activation chunk */
  for (BinNum b = 0; b < kNumBins; b++) {
    size_t bin_size = BinNumToSize(b);
    VLOG(1) << "Creating bin of max chunk size " << strings::HumanReadableNumBytes(bin_size);
    bool sharable_ = true;
    new (BinFromIndex(b, sharable_)) Bin(this, bin_size);
    CHECK_EQ(BinForSize(bin_size, sharable_), BinFromIndex(b, sharable_));
    CHECK_EQ(BinForSize(bin_size + 255, sharable_), BinFromIndex(b, sharable_));
    CHECK_EQ(BinForSize(bin_size * 2 - 1, sharable_),
             BinFromIndex(b, sharable_));
    if (b + 1 < kNumBins) {
      CHECK_NE(BinForSize(bin_size * 2, sharable_), BinFromIndex(b, sharable_));
    }
  }

  /* Bin for non-activation chunk */
  for (BinNum b = 0; b < kNumBins; b++) {
    size_t bin_size = BinNumToSize(b);
    VLOG(1) << "Creating bin of max chunk size " << strings::HumanReadableNumBytes(bin_size);
    bool sharable_ = false;
    new (BinFromIndex(b, sharable_)) Bin(this, bin_size);
    CHECK_EQ(BinForSize(bin_size, sharable_), BinFromIndex(b, sharable_));
    CHECK_EQ(BinForSize(bin_size + 255, sharable_), BinFromIndex(b, sharable_));
    CHECK_EQ(BinForSize(bin_size * 2 - 1, sharable_),
             BinFromIndex(b, sharable_));
    if (b + 1 < kNumBins) {
      CHECK_NE(BinForSize(bin_size * 2, sharable_), BinFromIndex(b, sharable_));
    }
  }
  std::cout << "BFCRecord,StepId,FunctionName,Allocator,ExecutorImplId,Allocationid,ChunkSize,"
            << "IndividualChunkLevelBytes,AggChunkLevelBytes,"
               "IndividualRegionLevelBytes,AggRegionLevelBytes,"
            << "Time,IsGradient,Sharable,SetOutput,IsBackward,LifeSpan,Caller,KernelName"
            << std::endl;
}

BFCAllocator::~BFCAllocator() {
  // Return memory back.
  VLOG(2) << "Number of regions allocated: "
          << region_manager_.regions().size();
  for (const auto& region : region_manager_.regions()) {
    if(debug_print == true){
      printf("---- ~BFCAllocator is calling sub_allocator_->Free(region.ptr(), %zu)\n", region.memory_size());
      fflush(stdout);
    }
    sub_allocator_mtx.lock();
    cr_global_pool_.RecordCUDA(name_, current_step_id, region.memory_size(), true, is_backward, "Free");
    sub_allocator_->Free(region.ptr(), region.memory_size());
    // cr_global_pool_.RecordCUDAFree(name_, current_step_id, region.memory_size(), true, is_backward);
    sub_allocator_mtx.unlock();
  }

	// Gangmuk: Bin separation
  for (BinNum b = 0; b < kNumBins; b++) {
    BinFromIndex(b, true)->~Bin();
		BinFromIndex(b, false)->~Bin();
		// BinFromIndex(b)->~Bin();
  }
  cr_global_pool_.num_allocators--; // Corresponding BFCStatus instance destructor is required.
}

void BFCAllocator::PrintOverhead(string caller, uint64 start_time) {
  if (!new_step) {
    if (print_bfc_overhead_step > 0 && (current_step_id >= print_bfc_step+10 && current_step_id < print_bfc_step+20)) {
      uint64 end_time = Env::Default()->NowNanos();
      uint64 elapsed = end_time - start_time;
      std::tuple<int, int, uint64, uint64> tup = std::make_tuple(current_step_id, gettid(), start_time, end_time);
      if (caller == "AllocOutside") {
        sum_overhead_allocoutside += elapsed;
        // alloc_outer_vec.push_back(elapsed);
        // alloc_outer_pair_vec.push_back(std::make_pair(start_time, end_time));
        alloc_outer_tuple_vec.push_back(tup);
        // count_allocoutside += 1;
      }
      else if (caller == "AllocInside") {
        sum_overhead_allocinside += elapsed;
        // alloc_inner_vec.push_back(elapsed);
        // alloc_inner_pair_vec.push_back(std::make_pair(start_time, end_time));
        alloc_inner_tuple_vec.push_back(tup);
        // count_allocinside += 1;
      }
      else if (caller == "FindChunkPtr") {
        sum_overhead_findchunkptr += elapsed;
        // findchunkptr_vec.push_back(elapsed);
        // findchunkptr_pair_vec.push_back(std::make_pair(start_time, end_time));
        findchunkptr_tuple_vec.push_back(tup);
        // count_findchunkptr += 1;
      }
      else if (caller == "RegionInit") {
        sum_overhead_regioninit += elapsed;
        // regioninit_vec.push_back(elapsed);
        // regioninit_pair_vec.push_back(std::make_pair(start_time, end_time));
        regioninit_tuple_vec.push_back(tup);
        // count_regioninit += 1;
      }
      else if (caller == "RegionInit_findentry") {
        sum_overhead_regioninit_findentry += start_time; // CAUTION!!!! start_time is elapsed time here.
        // regioninit_findentry_pair_vec.push_back(std::make_pair(start_time, end_time));
        // regioninit_findentry_tuple_vec.push_back(tup);
        // count_regioninit_findentry += 1;
      }
      else if (caller == "RegionInit_insertion") {
        sum_overhead_regioninit_insertion += start_time; // CAUTION!!!! start_time is elapsed time here.
        // regioninit_insertion_pair_vec.push_back(std::make_pair(start_time, end_time));
        // regioninit_insertion_tuple_vec.push_back(tup);
        // count_regioninit_insertion += 1;
      }
      else if (caller == "RegionAlloc") {
        sum_overhead_regionalloc += elapsed;
        // regionalloc_vec.push_back(elapsed);
        // regionalloc_pair_vec.push_back(std::make_pair(start_time, end_time));
        regionalloc_tuple_vec.push_back(tup);
        // count_regionalloc += 1;
      }
      else if (caller == "DeallocOutside") {
        sum_overhead_deallocoutside += elapsed;
        // dealloc_outer_vec.push_back(elapsed);
        // dealloc_outer_pair_vec.push_back(std::make_pair(start_time, end_time));
        dealloc_outer_tuple_vec.push_back(tup);
        // count_deallocoutside += 1;
      }
      else if (caller == "DeallocInside") {
        sum_overhead_deallocinside += elapsed;
        // dealloc_inner_vec.push_back(elapsed);
        // dealloc_inner_pair_vec.push_back(std::make_pair(start_time, end_time));
        dealloc_inner_tuple_vec.push_back(tup);
        // count_deallocinside += 1;
      }
      else if (caller == "RegionFree") {
        sum_overhead_regionfree += elapsed;
        // regionfree_vec.push_back(elapsed);
        // regionfree_pair_vec.push_back(std::make_pair(start_time, end_time));
        regionfree_tuple_vec.push_back(tup);
        // count_regionfree += 1;
      }
    }
  }
}


BFCAllocator::Chunk* BFCAllocator::ChunkFromHandle(ChunkHandle h) {
  DCHECK_GE(h, 0);
  DCHECK_LT(h, static_cast<int>(chunks_.size()));
  return &(chunks_[h]);
}

void BFCAllocator::RegisterNewAllocationRegion(
    size_t bytes, void* mem_addr, size_t region_id,
    std::map<std::string, std::string> node_info, bool from_gp, ChunkHandle* handles) {
  // Declaration to make consistent variable naming with sub_alloc part
  total_region_allocated_bytes_ += bytes; // total_region_allocated_bytes_ 업데이트 되는 지점.
  bool sharable_ = GetSharableFromNodeInfo(node_info);
  if (sharable_) {
    total_sharable_region_allocated_bytes_ += bytes;
  } else {
    total_nonsharable_region_allocated_bytes_ += bytes;
  }
  // const uint64 register_region_start_time = Env::Default()->NowNanos();
  std::pair<uint64, uint64> times = region_manager_.AddAllocationRegion(mem_addr, bytes, region_id, sharable_, from_gp, handles);
  // AllocationRegion* AR_ptr = const_cast<AllocationRegion*>(region_manager_.RegionFor(mem_addr));
  // AR_ptr->birth_time = Env::Default()->NowNanos();
  // printf("AR_ptr->birth_time,%llu\n", AR_ptr->birth_time);
  
  // PrintOverhead("RegionInit", register_region_start_time);
  // PrintOverhead("RegionInit_findentry", times.first);
  // PrintOverhead("RegionInit_insertion", times.second);
  // Create one large chunk for the whole memory space that will
  // be chunked later.
  ChunkHandle h = AllocateChunk();
  BFCAllocator::Chunk* c = ChunkFromHandle(h);
  // this is the address we have to memorize for later cuMemFree()
  c->ptr = mem_addr;
  c->size = bytes;
  c->allocation_id = -1;
  c->gangmuk_allocation_id = -1;
  c->prev = kInvalidChunkHandle;
  c->next = kInvalidChunkHandle;
  c->sharable = sharable_;  // Gangmuk: ACT
  CHECK(h != kInvalidChunkHandle);  // valid chunk's handle can't be set to kInvalidChunkHandle.
  region_manager_.set_handle(c->ptr, h);
  // Insert the chunk into the right bin.
  InsertFreeChunkIntoBin(h, sharable_);
}

// Gangmuk: ACT
// with GlobalPool coordination
int BFCAllocator::Extend(size_t alignment, size_t rounded_bytes, std::map<std::string, std::string> node_info) {
  if (debug_print) {
    printf("++ Extend with GlobalPool coordination.\n");
    fflush(stdout);
  }
  size_t available_bytes = memory_limit_ - total_region_allocated_bytes_;
  // Rounds available_bytes down to the nearest multiple of kMinAllocationSize.
  available_bytes = (available_bytes / kMinAllocationSize) * kMinAllocationSize;

  // Do we have enough space to handle the client's request?
  // If not, fail immediately.
  if (rounded_bytes > available_bytes) {
    if (debug_print) {
      printf("!! FAIL: Extend -- available_bytes: %d, round_bytes: %d\n", rounded_bytes, available_bytes);
      fflush(stdout);
    }
    return -1;
  }

  // 현재는 adaptive alloc에서 activation AR 인지 non-activation AR 인지 구분하지 않고 있다. 
  // 나중에 해야됨. => 이제 되어있는 듯.
  /*
    activate_global_pool 이고 phase two 이면 무조건 global pool에서 AR을 가져오려고 하고
    global_pool에서 가져오는 걸 실패하거나,
    activate_global_pool 이 안켜져 있거나, 아직 phase one 이면 GPU에서 추가 메모리를 할당 받는다.
  */
  bool sharable_ = GetSharableFromNodeInfo(node_info);
  // if (Name() != "cuda_host_bfc" && activate_global_pool &&
  //     (global_pool_release_all || (global_pool_release_ar && sharable_))) {
  if (Name() != "cuda_host_bfc" && activate_global_pool && (sharable_ || release_all)) {
    if (!adaptive_alloc || (adaptive_alloc && current_step_id >= start_aligning_step)) {
      if (debug_print) {
        printf("Try to get Region from GlobalPool\n");
        fflush(stdout);
      }
      Region ret_free_ar;
      // if (lock_global_pool) global_pool_mtx.lock();
      unsigned long long elapsed = Env::Default()->NowMicros() - base_micro;
      if (debug_print) {
        printf("before AllocRegionFromGlobalPool\n");
        fflush(stdout);
      }
      size_t min_required_size = default_ar_size;
      while (rounded_bytes > min_required_size) {
        if (tighten_ar) {
          size_t increasing_unit = 256 << 20; // 256MB
          min_required_size += increasing_unit;
        }
        else {
          min_required_size *= 2;
        }
      }
      // const uint64 alloc_region_start_time = Env::Default()->NowNanos();
      ret_free_ar = cr_global_pool_.AllocRegionFromGlobalPool(
          rounded_bytes, min_required_size, name_, current_step_id, elapsed,
          sharable_, total_region_allocated_bytes_, bfc_id);
      // PrintOverhead("RegionAlloc", alloc_region_start_time);
      if (debug_print) {
        printf("after AllocRegionFromGlobalPool\n");
        fflush(stdout);
      }
      // if (lock_global_pool) global_pool_mtx.unlock();
      if (ret_free_ar.memory_ptr != nullptr) {
        size_t bytes = ret_free_ar.size;
        void* mem_addr = ret_free_ar.memory_ptr;
        size_t region_id = ret_free_ar.region_id;
        bool from_gp = true;
        if (debug_print) {
          printf("after RegisterNewAllocationRegion\n");
          fflush(stdout);
        }
        RegisterNewAllocationRegion(bytes, mem_addr, region_id, node_info, from_gp, ret_free_ar.handles);
        if (debug_print) {
          printf("after RegisterNewAllocationRegion\n");
          fflush(stdout);
        }
        return 1;
      }
      else {
        printf("FailToGetRegionFromGlobalPool\n");
      }
    }
  }

  // 이미 여기 도달 했다는 건 이유가 어찌되었든 간에 GlobalPool을 통해서 AR을 Alloc하는 걸 실패했다는 뜻이기 때문에 
  // 조건문 확인 없이 바로 Alloc from GPU.
  /* 
    글로벌 풀 플래그가 켜져있지 않거나,
    글로벌 풀 플래그가 켜져있는데, sharable이 아니거나,
    글로벌 풀 플래그가 켜져있고 sharable인데, 글로벌 풀에서 AllocationRegion을 가져오지 못했으면,
    서브 얼로케이터를 통해 GPU로부터 메모리를 가져온다.
  */

  
  if (!tighten_ar) {
    curr_region_allocation_bytes_ *= 2;
    while (rounded_bytes > curr_region_allocation_bytes_) {
      curr_region_allocation_bytes_ *= 2;
    }
  }
  else {
    if (current_step_id < start_aligning_step - 2) {
      curr_region_allocation_bytes_ = 256 << 20;
      while (rounded_bytes > curr_region_allocation_bytes_) {
        curr_region_allocation_bytes_ += 256 << 20;
      }
      BFCAllocator::global_max_chunk_size = std::max(BFCAllocator::global_max_chunk_size, curr_region_allocation_bytes_);
      local_max_chunk_size = std::max(local_max_chunk_size, curr_region_allocation_bytes_);
    }
    else {
      // default_ar_size = BFCAllocator::global_max_chunk_size;
      // default_ar_size = local_max_chunk_size;
      curr_region_allocation_bytes_ = default_ar_size;
      while (rounded_bytes > curr_region_allocation_bytes_) {
        // curr_region_allocation_bytes_ += default_ar_size;
        curr_region_allocation_bytes_ += 256 << 20;
      }
    }
  }


  // // if (!sharable_ && current_step_id >= start_aligning_step && Name().find("GPU") != std::string::npos) {
  // if (GetSharableFromNodeInfo(node_info) == false) {
  //   if (current_step_id >= 10) {
  //     CHECK_EQ(num_nonsharable_region,1);
  //     CHECK_NE(nonsharable_chunk_peak_bytes,0);
  //     curr_region_allocation_bytes_ = nonsharable_chunk_peak_bytes + (16 << 20);
  //     printf("%s,%d,NonSharableRegionExtend,curr_region_allocation_bytes_,%zu\n", Name().c_str(), current_step_id, curr_region_allocation_bytes_);
  //     num_nonsharable_region++;
  //   }
  //   else {
  //     printf("%s,%d,SharableRegionExtend,curr_region_allocation_bytes_,%zu (NonSharable but current_step_id < 10)\n", Name().c_str(), current_step_id, curr_region_allocation_bytes_);
  //   }
  // }
  // else {
  //   printf("%s,%d,SharableRegionExtend,curr_region_allocation_bytes_,%zu\n", Name().c_str(), current_step_id, curr_region_allocation_bytes_);
  // }

  // Try allocating.
  size_t bytes = std::min(curr_region_allocation_bytes_, available_bytes);
  if (tighten_ar) {
    if (curr_region_allocation_bytes_ > available_bytes) {
      bytes = 256 << 20;
      while (rounded_bytes > bytes) {
        bytes += 256 << 20;
      }
    }
  }

  if ((long long)bytes < 0) {
    printf(" Error: bytes(%lld) < 0\n", (long long)bytes);
    exit(-1);
  }
  sub_allocator_mtx.lock();
  cr_global_pool_.RecordCUDA(name_, current_step_id, bytes, GetSharableFromNodeInfo(node_info), is_backward, "Alloc");
  void* mem_addr = sub_allocator_->Alloc(alignment, bytes);
  sub_allocator_mtx.unlock();
  // cr_global_pool_.RecordCUDAAlloc(name_, current_step_id, bytes, sharable_, is_backward);
  if (mem_addr == nullptr && !started_backpedal_) {
    printf("@@ mem_addr == nullptr => started_backpedal_\n");
    // Only backpedal once.
    started_backpedal_ = true;
    static constexpr float kBackpedalFactor = 0.9;
    // Try allocating less memory.
    while (mem_addr == nullptr) {
      bytes = RoundedBytes(bytes * kBackpedalFactor);
      if (bytes < rounded_bytes) break;
      sub_allocator_mtx.lock();
      cr_global_pool_.RecordCUDA(name_, current_step_id, bytes, sharable_, is_backward, "Alloc");
      mem_addr = sub_allocator_->Alloc(alignment, bytes);
      // cr_global_pool_.RecordCUDAAlloc(name_, current_step_id, bytes, sharable_, is_backward);
      sub_allocator_mtx.unlock();
    }
  }
  if (mem_addr == nullptr) {
    printf("!! FAIL: Extend -- mem_addr = nullptr...\n");
    return -1;
  }
  else {
    size_t region_id = -1;
    bool from_gp = false;
    RegisterNewAllocationRegion(bytes, mem_addr, region_id, node_info, from_gp, nullptr);
    
    VLOG(1) << "Extending allocation by " << strings::HumanReadableNumBytes(bytes)
            << " bytes.";
    VLOG(1) << "Total allocated bytes: "
            << strings::HumanReadableNumBytes(total_region_allocated_bytes_);
    VLOG(1) << "Allocated memory at " << mem_addr << " to "
            << static_cast<void*>(static_cast<char*>(mem_addr) + bytes);
    return 2;
  }
  printf("[GANGMUK_ERROR]: Extend shouldn't have been reached this point\n");
  fflush(stdout);
}

BFCAllocator::ChunkHandle BFCAllocator::AllocateChunk() {
  if (free_chunks_list_ != kInvalidChunkHandle) {
    ChunkHandle h = free_chunks_list_;
    Chunk* c = ChunkFromHandle(h);
    free_chunks_list_ = c->next;
		if (debug_print) {
      printf("++ AllocateChunk() returns chunk handle %zu.\n", h);
      fflush(stdout);
    }
    return h;
  } else {
    ChunkHandle h = chunks_.size();
    chunks_.resize(h + 1);
    return h;
  }
}

// Push this chunk to the head of BFC's free_chunks_list_
void BFCAllocator::DeallocateChunk(ChunkHandle h) {
	if (debug_print) {
    printf("-- DeallocateChunk(chunk[%zu]): Push this chunk to the head of BFC's free_chunks_list_\n", h);
    fflush(stdout);
  }
  Chunk* c = ChunkFromHandle(h);
  c->next = free_chunks_list_;
  free_chunks_list_ = h;
}

void* BFCAllocator::AllocateRaw(size_t unused_alignment, size_t num_bytes) {
  // Fast path: Try once to allocate without getting the retry_helper_ involved
  if(debug_print == true){
    printf("++ 2 arguments - AllocateRaw(bytes: %d) is calling AllocateRawInternal(bytes: %d)\n", num_bytes, num_bytes);
    fflush(stdout);
  }
  // point!
  void* r = AllocateRawInternal(unused_alignment, num_bytes, false);
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(r);
  Chunk* chunk = ChunkFromHandle(h);
  if (r != nullptr) {
    return r;
  } else {
    static const int64 kMaxMillisToWait = 10000;  // 10 seconds
    return retry_helper_.AllocateRaw(
        [this](size_t a, size_t nb, bool v) {
          return AllocateRawInternal(a, nb, v);
        },
        kMaxMillisToWait, unused_alignment, num_bytes);
  }
}

// Gangmuk: Newly Added 3
void* BFCAllocator::AllocateRaw(size_t unused_alignment, size_t num_bytes, 
                                std::map< std::string, std::string > node_info) {
  // Fast path: Try once to allocate without getting the retry_helper_ involved

  // const uint64 start_time_nsecs = Env::Default()->NowNanos();
  void* r = AllocateRawInternal(unused_alignment, num_bytes, false, node_info);
  // uint64 elapsed = Env::Default()->NowNanos() - start_time_nsecs;
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(r);
  Chunk* chunk = ChunkFromHandle(h);
  if (r != nullptr) {
    return r;
  } 
	else {
    printf("aaaaaaaaaaaaaaaaaaa It shouldn't have been printed !!!! aaaaaaaaaaaaaaaaaaaaa\n");
    static const int64 kMaxMillisToWait = 10000;  // 10 seconds
    return retry_helper_.AllocateRaw(
        [this](size_t a, size_t nb, bool v) {
          return AllocateRawInternal(a, nb, v);
        },
        kMaxMillisToWait, unused_alignment, num_bytes);
  }
}


/* 3 argument AllocateRaw */

void* BFCAllocator::AllocateRaw(size_t unused_alignment, size_t num_bytes,
                                const AllocationAttributes& allocation_attr) {
  if (allocation_attr.no_retry_on_failure) {
    // Return immediately upon the first failure if this is for allocating an
    // optional scratch space.
    bool dump_log_on_failure = VLOG_IS_ON(2);
    if(debug_print == true){
      printf("++ 3 arguments - AllocateRaw(bytes: %d) - if(no_retry) is calling AllocateRawInternal(bytes: %d)\n", num_bytes, num_bytes);
      fflush(stdout);
    }

    void* result = AllocateRawInternal(unused_alignment, num_bytes, dump_log_on_failure);
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(result);
    Chunk* chunk = ChunkFromHandle(h);
    if (result == nullptr) {
      static std::atomic<int32> log_counter{0};
      int32 counter_value = log_counter.load(std::memory_order_relaxed);
      if (counter_value < 10) {
        log_counter.store(counter_value + 1, std::memory_order_relaxed);
        LOG(WARNING)
            << "Allocator (" << Name() << ") ran out of memory trying "
            << "to allocate " << strings::HumanReadableNumBytes(num_bytes)
            << ". The caller indicates that this is not a failure, but"
            << " may mean that there could be performance gains if more"
            << " memory were available.";
      }
    }
    return result;
  } else {
    if(debug_print == true) {
      printf("++ 3 arguments - AllocateRaw(bytes: %d) - else(yes_retry) is calling AllocateRawInternal(bytes: %d)\n", num_bytes, num_bytes);
      fflush(stdout);
    }
    return AllocateRaw(unused_alignment, num_bytes);
  }
}

// Gangmuk: Newly Added 3
// Entry function from "allocate_tensor" in op_kernel.cc to BFC
void* BFCAllocator::AllocateRaw(size_t unused_alignment, size_t num_bytes,
                                const AllocationAttributes& allocation_attr, 
                                std::map< std::string ,std::string > node_info) {

  if (allocation_attr.no_retry_on_failure) {
    // Return immediately upon the first failure if this is for allocating an
    // optional scratch space.
    bool dump_log_on_failure = VLOG_IS_ON(2);
    if(debug_print == true){
      printf("++ 3 arguments - AllocateRaw(bytes: %d) - if(no_retry) is calling AllocateRawInternal(bytes: %d)\n", num_bytes, num_bytes);
      fflush(stdout);
    }

    void*  result = AllocateRawInternal(unused_alignment, num_bytes, dump_log_on_failure, node_info);
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(result);
    Chunk* chunk = ChunkFromHandle(h);
    if (result == nullptr) {
      static std::atomic<int32> log_counter{0};
      int32 counter_value = log_counter.load(std::memory_order_relaxed);
      if (counter_value < 10) {
        log_counter.store(counter_value + 1, std::memory_order_relaxed);
        LOG(WARNING)
            << "Allocator (" << Name() << ") ran out of memory trying "
            << "to allocate " << strings::HumanReadableNumBytes(num_bytes)
            << ". The caller indicates that this is not a failure, but"
            << " may mean that there could be performance gains if more"
            << " memory were available.";
      }
    }
    return result;
  }
  else {
    if(debug_print == true) {
      printf("++ 3 arguments - AllocateRaw(bytes: %d) - else(yes_retry) is calling AllocateRawInternal(bytes: %d)\n", num_bytes, num_bytes);
      fflush(stdout);
    }
    return AllocateRaw(unused_alignment, num_bytes, node_info);
  }
}


// static
size_t BFCAllocator::RoundedBytes(size_t bytes) {
  size_t rounded_bytes =
      (kMinAllocationSize *
       ((bytes + kMinAllocationSize - 1) / kMinAllocationSize));
  DCHECK_EQ(size_t{0}, rounded_bytes % kMinAllocationSize);
  return rounded_bytes;
}

///*
//get millisecond time
//*/
int BFCAllocator::getMilliCount(){
  timeb tb;
  ftime(&tb);
  int nCount = tb.millitm + (tb.time & 0xfffff) * 1000;
  return nCount;
}

int BFCAllocator::getMilliSpan(int nTimeStart){
  int nSpan = getMilliCount() - nTimeStart;
  if(nSpan < 0)
  nSpan += 0x100000 * 1000;
  return nSpan;
}

int BFCAllocator::getTimeDiff(){
  int diff = getMilliCount() - baseTime;
  return diff;
}

/*
	get microsecond using <chrono> library
*/

void BFCAllocator::RecordNodeInfoToChunk(
    void* ptr, std::map<std::string, std::string> node_info) {
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  Chunk* c = ChunkFromHandle(h);
  c->caller = node_info["caller"];
  c->kernel_name = node_info["kernel_name"];
  c->set_output = false;
  c->sharable = GetSharableFromNodeInfo(node_info);
  c->is_gradient = GetIsGradientFromNodeInfo(node_info);
  c->birth_time = Env::Default()->NowMicros();
  // c->executorstate_id = std::stoi(node_info["executorstate_id"]);
  c->executorimpl_id = std::stoi(node_info["executorimpl_id"]);
  // 이게 Host 메모리를 엄청나게 잡아먹어서 OOM이 나게 만들었다. 왜일까?
  // c->is_backward = cr_global_pool_.bfc_status_vec[bfc_id].is_backward; 
  c->is_backward = is_backward;
}

// Original AllocateRawInternal
void* BFCAllocator::AllocateRawInternal(size_t unused_alignment,
                                        size_t num_bytes,
                                        bool dump_log_on_failure) {
  // const uint64 alloc_outside_start_time = Env::Default()->NowNanos();
  // if (first_time_flag == true) {
  //   first_time_flag = false;
  //   baseTime = getMilliCount();
  // }
  if (num_bytes == 0) {
    LOG(ERROR) << "tried to allocate 0 bytes";
    return nullptr;
  }
  // First, always allocate memory of at least kMinAllocationSize
  // bytes, and always allocate multiples of kMinAllocationSize bytes
  // so all memory addresses are nicely byte aligned.
  size_t rounded_bytes = RoundedBytes(num_bytes);

  // The BFC allocator tries to find the best fit first.
  BinNum bin_num = BinNumForSize(rounded_bytes);

  /* Allocate Lock Point 1 */
  mutex_lock l(lock_);
  // const uint64 alloc_inside_start_time = Env::Default()->NowNanos();
  // mutex_lock l(lock_, 2); 
  // 1st trial FindingChunkPtr
  if (debug_print){
    printf("\n++ AllocateRawInternal(original-1st) is calling FindChunkPtr[%d, %d, %d]\n", bin_num, rounded_bytes, num_bytes);
    fflush(stdout);
  }
  std::map<std::string, std::string> dummy_node_info;
  dummy_node_info["kernel_name"] = "original-allocate_raw_internal-1st";
  dummy_node_info["op_type"] = "dummy";
  dummy_node_info["caller"] = "dummy";
  dummy_node_info["node_type"] = "dummy";
  dummy_node_info["is_gradient"] = "-1";
  dummy_node_info["sharable"] = "-1";
  dummy_node_info["step_id"] = "-1";
  dummy_node_info["node_id"] = "-1";
  // const uint64 findchunkptr_start_time = Env::Default()->NowNanos();
  void* ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, dummy_node_info);
  // PrintOverhead("FindChunkPtr", findchunkptr_start_time);
  if (debug_print){
    printf("  ++ After finishing 1st FindChunkPtr[%d, %d, %d]\n", bin_num, rounded_bytes, num_bytes);
    fflush(stdout);
  }
  if (ptr != nullptr) {
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
    const AllocationRegion* AR_ptr = region_manager_.RegionFor(ptr);
    size_t size_id = (AR_ptr->memory_size())/(b_to_mb);
    size_t region_id = AR_ptr->region_id();
    /////////////////////////////////////////////////////////////////
    CHECK(h != kInvalidChunkHandle);
    /////////////////////////////////////////////////////////////////
    Chunk* temp_chunk = ChunkFromHandle(h);
    size_t chunk_size = temp_chunk->size;
    // PrintOverhead("AllocInside", alloc_inside_start_time);
    // PrintOverhead("AllocOutside", alloc_outside_start_time);
    return ptr;
  }

  // Try to extend
  // If 1st trial fails, Extend the bfc memory pool by getting additional memory from system
  if (debug_print){
    printf("!! After FAIL to 1st FindChunkPtr, calling Extend...\n");
    fflush(stdout);
  }
  // const uint64 extend_start_time = Env::Default()->NowNanos();
  if (Extend(unused_alignment, rounded_bytes, dummy_node_info) > 0) {
    // PrintOverhead("Extend", extend_start_time);
    // 2nd trial FindingChunkPtr
    if (debug_print){
      printf("++ AllocateRawInternal(original-2nd) is calling FindChunkPtr...\n");
      fflush(stdout);
    }

    std::map<std::string, std::string> dummy_node_info;
    dummy_node_info["kernel_name"] = "original-allocate_raw_internal-2nd";
    dummy_node_info["op_type"] = "dummy";
    dummy_node_info["caller"] = "dummy";
    dummy_node_info["node_type"] = "dummy";
    dummy_node_info["is_gradient"] = "-1";
    dummy_node_info["sharable"] = "-1";
    dummy_node_info["step_id"] = "-1";
    dummy_node_info["node_id"] = "-1";
    // const uint64 findchunkptr_start_time = Env::Default()->NowNanos();
    ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, dummy_node_info);
    // PrintOverhead("FindChunkPtr", findchunkptr_start_time);
    if (debug_print){
      printf("  ++ After finishing 2nd FindChunkPtr[%d, %d, %d]\n", bin_num, rounded_bytes, num_bytes);
      fflush(stdout);
    }
    if (ptr != nullptr) {
      // PrintOverhead("AllocInside", alloc_inside_start_time);
      // PrintOverhead("AllocOutside", alloc_outside_start_time);
      return ptr;
    }
    else {
      if (debug_print){
        printf("!! AllocateRawInternal-original => FAIL: 2nd - FindChunkPtr return nullptr...\n");
        fflush(stdout);
      }
    }
  }

  // This is the case of failure even after Extending BFC pool.

  // We searched all bins for an existing free chunk to use and
  // couldn't find one.  This means we must have run out of memory,
  // Dump the memory log for analysis.
  if (dump_log_on_failure) {
    LOG(WARNING) << "Allocator (" << Name() << ") ran out of memory trying "
                 << "to allocate " << strings::HumanReadableNumBytes(num_bytes)
                 << ".  Current allocation summary follows.";
    DumpMemoryLog(rounded_bytes, false);
    LOG(WARNING) << RenderOccupancy();
  }
  return nullptr;
}

// ACT
// Gangmuk: Newly Added 3
// 3rd, third
void* BFCAllocator::AllocateRawInternal(
    size_t unused_alignment, size_t num_bytes, bool dump_log_on_failure,
    std::map<std::string, std::string> node_info) {
  // const uint64 alloc_outside_start_time = Env::Default()->NowNanos();
  if (num_bytes == 0) {
    LOG(ERROR) << "tried to allocate 0 bytes";
    return nullptr;
  }

  // First, always allocate memory of at least kMinAllocationSize
  // bytes, and always allocate multiples of kMinAllocationSize bytes
  // so all memory addresses are nicely byte aligned.
  size_t rounded_bytes = RoundedBytes(num_bytes);

  // The BFC allocator tries to find the best fit first.
  BinNum bin_num = BinNumForSize(rounded_bytes);

  mutex_lock l(lock_);

  int step_id = std::stoi(node_info["step_id"]);
  int executorimpl_id = std::stoi(node_info["executorimpl_id"]);
  if (Name().find("GPU") != std::string::npos) {
    new_step = IsNewStep(step_id, executorimpl_id);
    if (new_step) {
      cr_global_pool_.bfc_status_vec[bfc_id].EndOfOldStep(current_step_id);
      if (print_bfc == 1 && cr_global_pool_.bfc_status_vec[bfc_id].bfclog_vec.size() > 0) {
        if (step_id % 1 == 0) {
          mutex_lock l(cr_global_pool_.lock_);
          cr_global_pool_.bfc_status_vec[bfc_id].BFCLogFileWrite();
        }
      }
      if (debug_print) {
        printf("Return from EndOfOldStep\n");
        fflush(stdout);
      }
      if (step_id >= start_aligning_step) {
        if (cr_global_pool_.ModelBasedCoordination()) {
          cr_global_pool_.SimulateIfNewStep(bfc_id, current_step_id);
        }
      }
      if (debug_print) {
        printf("Return from SimulateIfNewStep\n");
        fflush(stdout);
      }
      cr_global_pool_.bfc_status_vec[bfc_id].ActualStartOfNewStep();
    }
  }

  // const uint64 alloc_inside_start_time = Env::Default()->NowNanos();
  if (debug_print) {
    printf("newly added 3 AllocateRawInternal\n");
    fflush(stdout);
  }
  int64 next_id = gangmuk_next_allocation_id_ + 1;
  IsBackward(next_id);

  // FindingChunkPtr 1st Trial
	// Gangmuk: ACT
  // const uint64 findchunkptr_start_time = Env::Default()->NowNanos();
  void* ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, node_info);
  // PrintOverhead("FindChunkPtr", findchunkptr_start_time);
  if (ptr != nullptr) {
    RecordNodeInfoToChunk(ptr, node_info);
    BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
    Chunk* c = ChunkFromHandle(h);
    total_chunk_allocated_bytes_ += c->size;
    UpdateChunkPeak(total_chunk_allocated_bytes_, c->gangmuk_allocation_id);
    UpdateRegionPeak(total_region_allocated_bytes_, c->gangmuk_allocation_id);
    if (Name().find("GPU") != std::string::npos) {
      BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
      Chunk* c = ChunkFromHandle(h);
      // cr_global_pool_.RecordAllocationInfo(current_step_id, bfc_id, c->allocation_id, c->size, c->is_gradient, c->sharable, c->set_output, stats_.bytes_in_use, total_region_allocated_bytes_, node_info);
      Chunk_MetaData metadata;
      metadata.set_output = c->set_output;
      metadata.sharable = c->sharable;
      metadata.is_gradient = c->is_gradient;
      metadata.lifespan = 0;
      metadata.allocation_id = c->allocation_id;
      metadata.gangmuk_allocation_id = c->gangmuk_allocation_id;
      metadata.chunk_size = c->size;
      metadata.caller = c->caller;
      metadata.kernel_name = c->kernel_name;
      // metadata.executorstate_id = c->executorstate_id;
      metadata.executorimpl_id = c->executorimpl_id;
      metadata.is_backward = c->is_backward;
      cr_global_pool_.RecordMemOperation(current_step_id, bfc_id, 1, metadata, stats_.bytes_in_use, total_region_allocated_bytes_);
    }
    if (debug_print) {
      printf("++ Success new3-1st FindChunkPtr[bin_num: %d, rounded_bytes: %d,num_bytes: %d, sharable: %s]\n",
              bin_num, rounded_bytes, num_bytes,
              SharableToString(std::stoi(node_info["sharable"])).c_str());
      fflush(stdout);
    }
    // PrintOverhead("AllocInside", alloc_inside_start_time);
    // PrintOverhead("AllocOutside", alloc_outside_start_time);
    new_step = false;
    return ptr;
  }

  if (debug_print){
    printf("!! new3-1st FindChunkPtr FAIL!! -> calling Extend\n");
    fflush(stdout);
  }
  if (Extend(unused_alignment, rounded_bytes, node_info) > 0) {
    if (debug_print){
      printf("++ After Extend, AllocateRawInternal(new3-2nd) is trying FindChunkPtr\n");
      fflush(stdout);
    }
    // FindingChunkPtr 2nd Trial
		if (debug_print) {
			// FindChunkPtr,bin_num,rounded_bytes,sharable,start,end
			printf("FindChunkPtr-2nd,%d,%zu,%s\n", bin_num, rounded_bytes, SharableToString(GetSharableFromNodeInfo(node_info)).c_str());
      fflush(stdout);
		}
    // const uint64 findchunkptr_start_time = Env::Default()->NowNanos();
    ptr = FindChunkPtr(bin_num, rounded_bytes, num_bytes, node_info);
    // PrintOverhead("FindChunkPtr", findchunkptr_start_time);
    if (debug_print){
      printf("  ++ After finishing 2nd FindChunkPtr[%d, %d, %d]\n", bin_num, rounded_bytes, num_bytes);
      fflush(stdout);
    }
    if (ptr != nullptr) {
      RecordNodeInfoToChunk(ptr, node_info);
      BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
      Chunk* c = ChunkFromHandle(h);
      // IsBackward(c->gangmuk_allocation_id);
      total_chunk_allocated_bytes_ += c->size;
      UpdateChunkPeak(total_chunk_allocated_bytes_, c->gangmuk_allocation_id);
      UpdateRegionPeak(total_region_allocated_bytes_, c->gangmuk_allocation_id);
      if (Name().find("GPU") != std::string::npos) {
        // BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
        // Chunk* c = ChunkFromHandle(h);
        // if (new_step) {
        //   cr_global_pool_.bfc_status_vec[bfc_id].EndOfOldStep(current_step_id);
        //   ///////////////////////////////////////////////////////////////////
        //   if (cr_global_pool_.ModelBasedCoordination()) {
        //     cr_global_pool_.SimulateIfNewStep(bfc_id, current_step_id);
        //   }
        //   ///////////////////////////////////////////////////////////////////
        //   cr_global_pool_.bfc_status_vec[bfc_id].ActualStartOfNewStep();
        // }
        // cr_global_pool_.RecordAllocationInfo(current_step_id, bfc_id, c->allocation_id, c->size, c->is_gradient, c->sharable, c->set_output, stats_.bytes_in_use, total_region_allocated_bytes_, node_info);
        Chunk_MetaData metadata;
        metadata.set_output = c->set_output;
        metadata.sharable = c->sharable;
        metadata.is_gradient = c->is_gradient;
        metadata.lifespan = 0;
        metadata.allocation_id = c->allocation_id;
        metadata.chunk_size = c->size;
        metadata.caller = c->caller;
        metadata.kernel_name = c->kernel_name;
        // metadata.executorstate_id = c->executorstate_id;
        metadata.executorimpl_id = c->executorimpl_id;
        cr_global_pool_.RecordMemOperation(current_step_id, bfc_id, 1, metadata, stats_.bytes_in_use, total_region_allocated_bytes_);
      }
      // PrintOverhead("AllocInside", alloc_inside_start_time);
      // PrintOverhead("AllocOutside", alloc_outside_start_time);
      new_step = false;
      return ptr;
    }
    if (debug_print){
      printf("!! FAIL: 2nd - FindChunkPtr return nullptr...\n");
      fflush(stdout);
    }
    return nullptr;
  }

  // This is the case of failure even after Extending BFC pool.

  // We searched all bins for an existing free chunk to use and
  // couldn't find one.  This means we must have run out of memory,
  // Dump the memory log for analysis.
  if (dump_log_on_failure) {
    LOG(WARNING) << "Allocator (" << Name() << ") ran out of memory trying "
                 << "to allocate " << strings::HumanReadableNumBytes(num_bytes)
                 << ".  Current allocation summary follows.";
    DumpMemoryLog(rounded_bytes, false);
    LOG(WARNING) << RenderOccupancy();
  }
  return nullptr;
}

// Gangmuk: ACT
void* BFCAllocator::FindChunkPtr(
    BinNum bin_num, size_t rounded_bytes, size_t num_bytes,
    std::map<std::string, std::string> node_info) {
  // const uint64 start_time_nsecs = Env::Default()->NowNanos();
  // First identify the first bin that could satisfy rounded_bytes.
  for (; bin_num < kNumBins; bin_num++) {
    // Start searching from the first bin for the smallest chunk that fits rounded_bytes.
		Bin* b = BinFromIndex(bin_num, GetSharableFromNodeInfo(node_info));
    /* Fine-Grained Lock Point */
    for (auto citer = b->free_chunks.begin(); citer != b->free_chunks.end(); ++citer) {
			//////////////////////////////////////////////////
      const BFCAllocator::ChunkHandle h = (*citer);
			if (h == kInvalidChunkHandle) {
				printf("@@@@ Chunk handle: %d\n", h);
        fflush(stdout);
			}
      CHECK(h != kInvalidChunkHandle); // Inserted by me
			//////////////////////////////////////////////////
      BFCAllocator::Chunk* chunk = ChunkFromHandle(h);
			//////////////////////////////////////////////////
			size_t temp_h = region_manager_.get_handle(chunk->ptr);
			if (temp_h == kInvalidChunkHandle) {
				printf("@@@@ h = (*citer): %d\n", h);
				printf("@@@@ region_manager_.get_handle(chunk->ptr): %d\n", temp_h);
				printf("@@@@ h == kInvalidChunkHandle\n");
        fflush(stdout);
			}
			CHECK(h != kInvalidChunkHandle); // Inserted by Gangmuk
			//////////////////////////////////////////////////

      DCHECK(!chunk->in_use());
      if (chunk->size >= rounded_bytes) {
				// Bin에 있는 free_chunk의 sharable 값과 요청 받은 sharable값은 같아야한다. 
        // 잘 작동하면 if 문은 항상 참이어야한다.
        // It must be always true
        CHECK_EQ(chunk->sharable, GetSharableFromNodeInfo(node_info));
				if (chunk->sharable == GetSharableFromNodeInfo(node_info)) {
					// We found an existing chunk that fits us that wasn't in use, so remove
					// it from the free bin structure prior to using.
					RemoveFreeChunkIterFromBin(&b->free_chunks, citer);

					if (debug_print) {
            printf("++ Find Chunk[%zu] size: %zu)\n", h, chunk->size);
            fflush(stdout);
          }

					// If we can break the size of the chunk into two reasonably large
					// pieces, do so.  In any case don't waste more than
					// kMaxInternalFragmentation bytes on padding this alloc.
					const int64 kMaxInternalFragmentation = 128 << 20;  // 128mb
					if (chunk->size >= rounded_bytes * 2 || static_cast<int64>(chunk->size) - rounded_bytes >= kMaxInternalFragmentation) {
						///////////////////////////////
						SplitChunk(h, rounded_bytes);
						///////////////////////////////
						chunk = ChunkFromHandle(h);  // Update chunk pointer in case it moved
					}

					// The requested size of the returned chunk is what the user
					// has allocated.
					chunk->requested_size = num_bytes;
					// Assign a unique id and increment the id counter, marking the
					// chunk as being in use.
					chunk->allocation_id = next_allocation_id_++;
					chunk->gangmuk_allocation_id = gangmuk_next_allocation_id_++;

					// Update stats.
					++stats_.num_allocs;
					stats_.bytes_in_use += chunk->size;
          // if (!(chunk->sharable)) {
          //   temp_nonsharable_chunk_bytes += chunk->size;
          //   temp_nonsharable_chunk_peak_bytes = std::max(temp_nonsharable_chunk_peak_bytes, temp_nonsharable_chunk_bytes);
          // }
					stats_.max_bytes_in_use = std::max(stats_.max_bytes_in_use, stats_.bytes_in_use);
					stats_.max_alloc_size = std::max<std::size_t>(stats_.max_alloc_size, chunk->size);

          VLOG(4) << "Returning: " << chunk->ptr;
					if (VLOG_IS_ON(4)) {
						LOG(INFO) << "A: " << RenderOccupancy();
					}
          if (debug_print) {
            printf("FindChunkPtr is about to return the valid ptr.\n");
            fflush(stdout);
          }
          return chunk->ptr;
        }
				else {
          if (!one_bin) {
            printf(
                "GANGMUK_ERROR: chunk->size >= rounded_bytes BUT found_chunk->sharable: %s != required_sharable: %s\n",
                SharableToString(chunk->sharable).c_str(),
                SharableToString(GetSharableFromNodeInfo(node_info)).c_str());
            fflush(stdout);
            CHECK_EQ(0,1);
          }
				}
			}
		}
	}
  return nullptr;
}

void BFCAllocator::SplitChunk(BFCAllocator::ChunkHandle h, size_t num_bytes) {
  // Allocate the new chunk before we do any ChunkFromHandle
  ChunkHandle h_new_chunk = AllocateChunk();

  Chunk* c = ChunkFromHandle(h);
  size_t original_size = c->size;
  if(debug_print){
    if(original_size < 0){
      printf("!!!!!!! original_size(%zu) < 0\n", original_size);
      fflush(stdout);
    }
  }
  CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));

  // Create a new chunk starting num_bytes after c
  BFCAllocator::Chunk* new_chunk = ChunkFromHandle(h_new_chunk);
  new_chunk->ptr = static_cast<void*>(static_cast<char*>(c->ptr) + num_bytes);
  new_chunk->size = c->size - num_bytes;
  new_chunk->sharable = c->sharable;

  if(h==kInvalidChunkHandle){
    printf("region_manager_.set_handle(c->ptr, &&& kInvalidChunkHandle)\n");
    fflush(stdout);
  }
  if (debug_print){
    const AllocationRegion* AR_1_ptr = region_manager_.RegionFor(c->ptr);
    const AllocationRegion* AR_2_ptr = region_manager_.RegionFor(new_chunk->ptr);
    size_t region_id_1 = AR_1_ptr->region_id();
    size_t region_id_2 = AR_2_ptr->region_id();
    printf("++ SplitChunk: AR[%d](random_id: %d) [ <Chunk[%d]> => <Chunk[%d]> + <Chunk[%d]> ], [ size: %zu => %zu + %zu ]\n"
            , region_id_1, AR_1_ptr->random_id(), h, h, h_new_chunk, original_size, num_bytes, new_chunk->size);
    if (region_id_1 != region_id_2){
      printf("++ SplitChunk: region_id_1[%d] != region_id_2[%d] !!!!!!!!!!!!!!!\n", region_id_1, region_id_2);
      fflush(stdout);
    }
  }

  if(debug_print){
    printf("++ SplitChunk() is calling region_manager_.set_handle(%d)\n", h_new_chunk);
    fflush(stdout);
  }
  region_manager_.set_handle(new_chunk->ptr, h_new_chunk);

  // Set the new sizes of the chunks.
  c->size = num_bytes;

  // The new chunk is not in use.
  new_chunk->allocation_id = -1;
  new_chunk->gangmuk_allocation_id = -1;

  // Maintain the pointers.
  // c <-> c_neighbor becomes
  // c <-> new_chunk <-> c_neighbor
  BFCAllocator::ChunkHandle h_neighbor = c->next;
  new_chunk->prev = h;
  new_chunk->next = h_neighbor;
  c->next = h_new_chunk;
  if (h_neighbor != kInvalidChunkHandle) {
    Chunk* c_neighbor = ChunkFromHandle(h_neighbor);
    c_neighbor->prev = h_new_chunk;
  }

  // Add the newly free chunk to the free bin.
  InsertFreeChunkIntoBin(h_new_chunk, c->sharable);
}

void BFCAllocator::DeallocateRaw(void* ptr) {
  if(debug_print == true) {
    printf("-- (bfc_allocator.cc) DeallocateRaw(ptr) is calling DeallocateRawInternal(ptr)\n");
    fflush(stdout);
  }
  // const uint64 start_time_nsecs = Env::Default()->NowNanos();

  DeallocateRawInternal(ptr);

  retry_helper_.NotifyDealloc();

  /////////////////////////////////////////////////////////////////////
  // 버그 유발 지점./////////////////////////////////////////////////////
  // Dealloc하고 해당 청크를 refer하려고 하면 안됩니다.
  // 특히나, Region을 완전히 프리한 DeallocateRaw이후에 과거에 거기에 있었던
  // chunk를 refer하려고 하면 그냥 바로 Segmentation Fault 입니다.
  // BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  // printf("after DeallocateRaw region_manager_.get_handle(ptr)\n");
  // fflush(stdout);
  // Chunk* chunk = ChunkFromHandle(h);
  // uint64 elapsed = Env::Default()->NowNanos() - start_time_nsecs;
  // PrintOverhead_1("DeallocOutsideOverhead", chunk->size, elapsed);
  /////////////////////////////////////////////////////////////////////

  if (debug_print) {
    printf("Finish DeallocateRaw\n");
    fflush(stdout);
  }
}

void BFCAllocator::DeallocateRawInternal(void* ptr) {
  // const uint64 dealloc_outside_start_time = Env::Default()->NowNanos();
  if (ptr == nullptr) {
    LOG(ERROR) << "tried to deallocate nullptr";
    return;
  }

  /* Deallocate Lock Point */
  mutex_lock l(lock_);
  // const uint64 dealloc_inside_start_time = Env::Default()->NowNanos();
  // const uint64 start_time_nsecs = Env::Default()->NowNanos();
  /* overloaded mutex_lock class ctor prints second parameter. This is for debugging when deadlock occurs */
  // mutex_lock l(lock_, 1); 
  
  // Find the chunk from the ptr.
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle);

  Chunk* c = ChunkFromHandle(h);
  const AllocationRegion* AR_ptr = region_manager_.RegionFor(c->ptr); 
  size_t region_id = AR_ptr->region_id();

  if (Name().find("GPU") != std::string::npos) {
    total_chunk_allocated_bytes_ -= c->size;
    unsigned long long lifespan = Env::Default()->NowMicros() - c->birth_time;
    // cr_global_pool_.RecordDeallocationInfo(current_step_id, bfc_id, c->allocation_id, c->size, c->is_gradient, c->sharable, c->set_output, lifespan, stats_.bytes_in_use, total_region_allocated_bytes_, c->kernel_name, c->caller);
    Chunk_MetaData metadata;
    metadata.set_output = c->set_output;
    metadata.sharable = c->sharable;
    metadata.is_gradient = c->is_gradient;
    metadata.lifespan = lifespan;
    metadata.allocation_id = c->allocation_id;
    metadata.gangmuk_allocation_id = c->gangmuk_allocation_id;
    metadata.chunk_size = c->size;
    metadata.caller = c->caller;
    metadata.kernel_name = c->kernel_name;
    // metadata.executorstate_id = c->executorstate_id;
    metadata.executorimpl_id = c->executorimpl_id;
    metadata.is_backward = c->is_backward;
    cr_global_pool_.RecordMemOperation(current_step_id, bfc_id, -1, metadata, stats_.bytes_in_use, total_region_allocated_bytes_);
  }

  FreeAndMaybeCoalesce(h);
  // uint64 elapsed = Env::Default()->NowNanos() - start_time_nsecs;
  // PrintOverhead_1("DeallocInsideOverhead", -1, elapsed);

  // 여기도 마찬가지 입니다. 버그 유발 지점 입니다.
  // FreeAndMaybeCoalesce 함수 안에서 Region Free가 일어나기 때문에.
  // chunk를 refer하려고 하면 그냥 바로 Segmentation Fault 입니다.
  // Chunk* chunk = ChunkFromHandle(h);
  // uint64 elapsed = Env::Default()->NowNanos() - start_time_nsecs;
  // PrintOverhead_1("DeallocInsideOverhead", chunk->size, elapsed);

  if (VLOG_IS_ON(4)) {
    LOG(INFO) << "F: " << RenderOccupancy();
  }
  if (debug_print) {
    printf("Finish DeallocateRawInternal\n");
    fflush(stdout);
  }
  // PrintOverhead("DeallocInside", dealloc_inside_start_time);
  // PrintOverhead("DeallocOutside", dealloc_outside_start_time);
}

/*
	Merges h1 and h2 when Chunk(h1)->next is h2 and Chunk(h2)->prev is c1.
	We merge Chunk(h2) into Chunk(h1).
	Return -1: Initial value
				  0: No CUDA FREE and No GlobalPool FREE
				  1: CUDA FREE
				  2: GlobalPool FREE
*/
int BFCAllocator::Merge(BFCAllocator::ChunkHandle h1,
                        BFCAllocator::ChunkHandle h2) {
  Chunk* c1 = ChunkFromHandle(h1);
  Chunk* c2 = ChunkFromHandle(h2);

  if(debug_print){
		const AllocationRegion* AR_1_ptr = region_manager_.RegionFor(c1->ptr);
		const AllocationRegion* AR_2_ptr = region_manager_.RegionFor(c2->ptr);
		size_t region_id_1 = AR_1_ptr->region_id();
		size_t region_id_2 = AR_2_ptr->region_id();
    printf("-- Merge: AR[%d]'s [ <Chunk[%d]> + <Chunk[%d]> => <Chunk[%d]> ]  [ size[%zu] + size[%zu] => size[%zu] ]\n",
            region_id_1, h1, h2, h1, c1->size, c2->size, c1->size+c2->size);
    fflush(stdout);
    if (region_id_1 != region_id_2){
      printf("-- Merge: region_id_1[%d] != region_id_2[%d]  !!!!!!!!!!!!!!\n", region_id_1, region_id_2);
      fflush(stdout);
    }
	}

  // We can only merge chunks that are not in use.
  CHECK(!c1->in_use() && !c2->in_use());
  // Gangmuk
  CHECK_EQ(c1->sharable, c2->sharable);

  // c1's prev doesn't change, still points to the same ptr, and is
  // still not in use.

  // Fix up neighbor pointers
  //
  // c1 <-> c2 <-> c3 should become
  // c1 <-> c3

  BFCAllocator::ChunkHandle h3 = c2->next;
  c1->next = h3;
  CHECK(c2->prev == h1);
  if (h3 != kInvalidChunkHandle) {
    BFCAllocator::Chunk* c3 = ChunkFromHandle(h3);
    c3->prev = h1;
  }

  // Set the new size
  c1->size += c2->size;

 	DeleteChunk(h2);
  return TryToFreeAllocationRegion(h1);
}

/*
	CORE FUNCTION: Free AllocationRegion, if neighbor chunks are InvalidChunk.
	'h' is our DeleteChunk target.
	'h1' is  'h''s neighbor(prev) chunk.
	This function returns 'true', when it frees out Allocation Region after DeleteChunk.

  h가 delete할 청크의 핸들이고 원래 함수는 h만 파라미터로 받았다.
  h1은 내가 새로 추가한 파라미터인데, h를 멀지(합병)시키는, 
  즉, h1은 h를 흡수하는 청크의 핸들이다.
*/
int BFCAllocator::DeleteChunk(ChunkHandle h) { //, BFCAllocator::ChunkHandle h1) {
  // Delete h and cleanup all state
  CHECK(h != kInvalidChunkHandle);
  Chunk* c = ChunkFromHandle(h);

  const AllocationRegion* AR_ptr = region_manager_.RegionFor(c->ptr);
  void* AR_memory_ptr = AR_ptr->ptr();
  size_t AR_size = AR_ptr->memory_size();
  size_t region_id = AR_ptr->region_id();
  if (debug_print) {
    printf("-- DeleteChunk: AR[%d]'s Chunk[%d]\n", region_id, h);
    fflush(stdout);
  }

  //  VLOG(4) << "Removing: " << c->ptr;

	/*
		지금 밑에 두줄이 원래 이 함수의 역할이다. 그리고 원래 DeleteChunk 함수의 전부임.
		그런데, 저걸 만약 AR을 프리할꺼면 안해야되는 게 맞는건가?
		DeallocateChunk를 하면 AR레벨이 아닌 BFC레벨에서 관리되고 있는 free_chunks_list_에청크 핸들을 넣을 텐데,
		free_chunks_list_에서 나중에 다른 chunk allocation을 할때 저 프리된 AR의 청크 핸들을 가져오면  문제가 되는 거 아닌가?
		확신은 없지만, 이거 확인해봐야됨.
		일단 여기 아니면, AR Free test 뒤로 밑에 두줄을 보내야되는거 같기도 하다. 해보자.
	*/
  // if (c != nullptr) {
    region_manager_.erase(c->ptr);
    DeallocateChunk(h);
}


int BFCAllocator::TryToFreeAllocationRegion(BFCAllocator::ChunkHandle h) {
  CHECK(h != kInvalidChunkHandle);
  Chunk* c = ChunkFromHandle(h);
  const AllocationRegion* AR_ptr = region_manager_.RegionFor(c->ptr);
  void* AR_memory_ptr = AR_ptr->ptr();
  size_t AR_size = AR_ptr->memory_size();
  size_t region_id = AR_ptr->region_id();
  // bool sharable_ = AR_ptr->sharable;
  // CHECK_EQ(AR_ptr->sharable, c->sharable);
  if (debug_print) {
    printf("TryToFreeAllocationRegion entry\n");
    fflush(stdout);
  }
  if (c->next != kInvalidChunkHandle || c->prev != kInvalidChunkHandle || Name() == "cuda_host_bfc") {
    return 0;
  }
  if (c->next == kInvalidChunkHandle && c->prev == kInvalidChunkHandle && Name() != "cuda_host_bfc") {
    /**** CUDA FREE STEP(Compulsory) ****/
    // if (cuda_free_step) {
    //   sub_allocator_mtx.lock();
    //   sub_allocator_->Free(AR_memory_ptr, AR_size);
    //   // cr_global_pool_.RecordCUDAFree(name_, current_step_id, AR_size, c->sharable, is_backward);
    //   cr_global_pool_.RecordCUDA(name_, current_step_id, AR_size, c->sharable, is_backward, "Free");
    //   sub_allocator_mtx.unlock();
    //   // last parameter is false if it is Non-Activation AllocationRegion
    //   delete[] AR_ptr->handles_;
    //   region_manager_.EraseAllocationRegion(AR_memory_ptr, region_id, false);  
    //   total_region_allocated_bytes_ -= AR_size; // total_region_allocated_bytes_ 업데이트 되는 지점.
    //   // Decrement curr_region_allocation_bytes_ after GPU release
    //   if (update_curr_region_allocation_size && curr_region_allocation_bytes_ - AR_size > 0) {
    //     curr_region_allocation_bytes_ -= AR_size;
    //   }
    //   if (debug_print) {
    //     printf(" --- finished to free AR --- total_region_allocated_bytes: %d\n", total_region_allocated_bytes_);
    //     fflush(stdout);
    //   }
    //   return 1;  // CUDA FREE
    // }

    /**** Adaptive alloc ****/
    if (adaptive_alloc) {
      /**** No activate global pool ****/

      /**** Activate global pool ****/
      if (activate_global_pool) {
        /**** Phase two ****/
        // if (phase_two_){
        if (current_step_id >= start_aligning_step){
          /**** Release sharable only ****/
          if (c->sharable || release_all) {
            unsigned long long elapsed = Env::Default()->NowMicros() - base_micro;
            // if (lock_global_pool) global_pool_mtx.lock();
            // 여기서의 bytes_in_use는 이미 update가 된 상태이다.
            // const uint64 free_region_start_time = Env::Default()->NowNanos();
            cr_global_pool_.FreeRegionToGlobalPool(
                AR_memory_ptr, AR_size, region_id, c->sharable, name_,
                current_step_id, elapsed, bfc_id, total_region_allocated_bytes_, AR_ptr->handles_, AR_ptr->birth_time);
            assert(AR_ptr->handles_ == nullptr);
            // PrintOverhead("RegionFree", free_region_start_time);
            // if (lock_global_pool) global_pool_mtx.unlock();
            region_manager_.EraseAllocationRegion(AR_memory_ptr, region_id, true);
            total_region_allocated_bytes_ -= AR_size; // total_region_allocated_bytes_ 업데이트 되는 지점.
            // Decrement curr_region_allocation_bytes_ after GPU release
            if (update_curr_region_allocation_size && curr_region_allocation_bytes_ - AR_size > 0) {
              curr_region_allocation_bytes_ -= AR_size;
            }
            return 2;  // GlobalPool Free
          }
          /**** Release sharable only But this reigon is non-sharable. ****/
          else {
            printf("NonSharableRegion,No GlobalPool Free,%zu\n", AR_size);
            return 0; // No Free
          }
        }
        /**** Phase one ****/
        // else {
        else if (current_step_id >= cuda_free_start_step) {
          sub_allocator_mtx.lock();
          cr_global_pool_.RecordCUDA(name_, current_step_id, AR_size, c->sharable, is_backward, "Free");
          sub_allocator_->Free(AR_memory_ptr, AR_size);
          // cr_global_pool_.RecordCUDAFree(name_, current_step_id, AR_size, c->sharable, is_backward);
          sub_allocator_mtx.unlock();
          // last parameter is false if it is Non-Activation AllocationRegion
          delete[] AR_ptr->handles_;
          region_manager_.EraseAllocationRegion( AR_memory_ptr, region_id, false);  
          total_region_allocated_bytes_ -= AR_size; // total_region_allocated_bytes_ 업데이트 되는 지점.
          // Decrement curr_region_allocation_bytes_ after GPU release
          if (update_curr_region_allocation_size && curr_region_allocation_bytes_ - AR_size > 0) {
            curr_region_allocation_bytes_ -= AR_size;
          }
          if (debug_print) {
            printf(" --- finished to free AR --- total_region_allocated_bytes: %d\n", total_region_allocated_bytes_);
            fflush(stdout);
          }
          return 1;  // CUDA FREE
        }
        else {
          return 0; // No Free
        }
      }
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      else {
        /**** Phase two ****/
        // if (phase_two_) {
        if (current_step_id >= start_aligning_step) {
          return 0; // No Free
        }
        /**** Phase one ****/
        // else {
        else if (current_step_id >= cuda_free_start_step) {
          sub_allocator_mtx.lock();
          cr_global_pool_.RecordCUDA(name_, current_step_id, AR_size, c->sharable, is_backward, "Free");
          sub_allocator_->Free(AR_memory_ptr, AR_size);
          // cr_global_pool_.RecordCUDAFree(name_, current_step_id, AR_size, c->sharable, is_backward);
          sub_allocator_mtx.unlock();
          // last parameter is false if it is Non-Activation AllocationRegion
          delete[] AR_ptr->handles_;
          region_manager_.EraseAllocationRegion( AR_memory_ptr, region_id, false);  
          total_region_allocated_bytes_ -= AR_size; // total_region_allocated_bytes_ 업데이트 되는 지점.
          // Decrement curr_region_allocation_bytes_ after GPU release
          if (update_curr_region_allocation_size && curr_region_allocation_bytes_ - AR_size > 0) {
            curr_region_allocation_bytes_ -= AR_size;
          }
          if (debug_print) {
            printf(" --- finished to free AR --- total_region_allocated_bytes: %d\n", total_region_allocated_bytes_);
            fflush(stdout);
          }
          return 1;  // CUDA FREE: // cuda_free_start_step보다 크고 start_aligning_step보다 작기 때문에 cuda free한다. .
        }
        else {
          return 0; // No Free: start_aligning_step보다 작은 스텝이긴 하지만 cuda_free_start_step에 아직 못 도달했기 때문에 free하지 않는다.
        }
      }

    }
    /**** No adaptive alloc ****/
    else {
      if (activate_global_pool) {
        /**** Release Sharable Only ****/
        // if (global_pool_release_ar) {
          /**** Release Sharable Only And This region is sharable. ****/
          if (c->sharable || release_all) {
            unsigned long long elapsed = Env::Default()->NowMicros() - base_micro;
            // if (lock_global_pool) global_pool_mtx.lock();
            // 여기서의 bytes_in_use는 이미 update가 된 상태이다.
            // const uint64 free_region_start_time = Env::Default()->NowNanos();
            cr_global_pool_.FreeRegionToGlobalPool(
                AR_memory_ptr, AR_size, region_id, c->sharable, name_,
                current_step_id, elapsed, bfc_id, total_region_allocated_bytes_, AR_ptr->handles_, AR_ptr->birth_time);
            assert(AR_ptr->handles_ == nullptr);
            // PrintOverhead("RegionFree", free_region_start_time);
            // if (lock_global_pool) global_pool_mtx.unlock();
            region_manager_.EraseAllocationRegion(AR_memory_ptr, region_id, true);
            total_region_allocated_bytes_ -= AR_size; // total_region_allocated_bytes_ 업데이트 되는 지점.
            // Decrement curr_region_allocation_bytes_ after GPU release
            if (update_curr_region_allocation_size && curr_region_allocation_bytes_ - AR_size > 0) {
              curr_region_allocation_bytes_ -= AR_size;
            }
            return 2;  // GlobalPool Free
          }
          /**** Release Sharable Only But This region is not sharable. ****/
          else {
            return 0; // No Free
          }
        // }
        // /**** Release All ****/
        // else if (global_pool_release_all) {
        //   unsigned long long elapsed = Env::Default()->NowMicros() - base_micro;
        //   if (lock_global_pool) global_pool_mtx.lock();
        //   // 여기서의 bytes_in_use는 이미 update가 된 상태이다.
        //   cr_global_pool_.FreeRegionToGlobalPool(
        //       AR_memory_ptr, AR_size, region_id, c->sharable, name_,
        //       current_step_id, elapsed, bfc_id, total_region_allocated_bytes_);
        //   if (lock_global_pool) global_pool_mtx.unlock();
        //   region_manager_.EraseAllocationRegion(AR_memory_ptr, region_id, true);
        //   total_region_allocated_bytes_ -= AR_size; // total_region_allocated_bytes_ 업데이트 되는 지점.
        //   // Decrement curr_region_allocation_bytes_ after GPU release
        //   if (update_curr_region_allocation_size && curr_region_allocation_bytes_ - AR_size > 0) {
        //     curr_region_allocation_bytes_ -= AR_size;
        //   }
        //   return 2;  // GlobalPool Free
        // } else {
        //   printf(
        //       "gangmuk_ERROR: You must turn on at least one out of "
        //       "global_pool_reelase_ar and global_pool_release_all for "
        //       "activate_global_pool.\n");
        //   CHECK_EQ(0, 1);
        // }
      }
      /**** No activate global pool ****/
      else {
        return 0; // No Free
      }
    }
  }

  /* This region is not free. */
  else {
    return 0; // No Free
  }
}

void BFCAllocator::InsertFreeChunkIntoBin(BFCAllocator::ChunkHandle h, bool sharable_) {
  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use() && (c->bin_num == kInvalidBinNum));
  if ((long long)c->size < 0) {
    printf("!!!!!!!! InsertFreeChunkIntoBin: c->size(%d) < 0\n", c->size);
    fflush(stdout);
  }

  BinNum bin_num = BinNumForSize(c->size);
  c->bin_num = bin_num;

	// Gangmuk: Bin separation
  Bin* new_bin = BinFromIndex(bin_num, sharable_);
  new_bin->free_chunks.insert(h);
	if (debug_print) {
    printf("++ Insert Free %s-Chunk[%zu] into Bin[%d]\n", SharableToString(sharable_).c_str(), h, bin_num);
    fflush(stdout);
  }
}


void BFCAllocator::RemoveFreeChunkIterFromBin(
    BFCAllocator::Bin::FreeChunkSet* free_chunks,
    const BFCAllocator::Bin::FreeChunkSet::iterator& citer) {
  ChunkHandle h = *citer;
  Chunk* c = ChunkFromHandle(h);
	/////////////////////////////////////////////////////
	/////////////////ERROR POINT/////////////////////////
	/////////////////////////////////////////////////////
  CHECK(!c->in_use() && (c->bin_num != kInvalidBinNum));

  free_chunks->erase(citer);
  c->bin_num = kInvalidBinNum;
}

void BFCAllocator::RemoveFreeChunkFromBin(BFCAllocator::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  CHECK(!c->in_use());
  CHECK(c->bin_num != kInvalidBinNum);

	// Gangmuk: Bin separation
	CHECK_GT(BinFromIndex(c->bin_num, c->sharable)->free_chunks.erase(h), 0)
			<< "Could not find chunk in bin";

  c->bin_num = kInvalidBinNum;
	if (debug_print) {
    printf("-- Remove Free Chunk From Bin (Chunk[%zu])\n",h);
    fflush(stdout);
  }
}

// Allocation Region will return to System when it is last merge
// e.g)
// AR[ c[i]-c[8]-c[9]-c[i] ] 
//  |
//  v
// AR[ c[i]-c[8]-c[i] ]
void BFCAllocator::FreeAndMaybeCoalesce(
    BFCAllocator::ChunkHandle h) {
  Chunk* c = ChunkFromHandle(h);
  CHECK(c->in_use());
  CHECK(c->bin_num == kInvalidBinNum);

  // Mark the chunk as no longer in use.
  c->allocation_id = -1;
  c->gangmuk_allocation_id = -1;

  // Updates the stats.
  stats_.bytes_in_use -= c->size;
  // if (!(c->sharable)) {
  //   temp_nonsharable_chunk_bytes -= c->size;
  // }

  ChunkHandle coalesced_chunk = h;

  Chunk* next_chunk = ChunkFromHandle(c->next);
  Chunk* prev_chunk = ChunkFromHandle(c->prev);

  int is_AR_released_1 = -1;
  int is_AR_released_2 = -1;
  
  /*
    Merge 조건 
      1) next 청크가 유효한 청크이고 (kInvalidChunkHandle이 아니고),  (유효하지 않은 청크와는 멀지를 하면 안되겠지?)
      2) next 청크가 사용중이 아니면 (in_use()가 false 이면, 즉 allocation_id가 -1 이면) (사용중인 청크와는 멀지를 하면 안되겠지?)
    1) 와 2) 조건이 모두 만족을 할 때만 Merge를 한다.
  */
  // If the next chunk is free, merge it into c and delete it.
  if (c->next != kInvalidChunkHandle && !ChunkFromHandle(c->next)->in_use()) {
    CHECK(region_manager_.RegionFor(c->ptr) == region_manager_.RegionFor(next_chunk->ptr));
    RemoveFreeChunkFromBin(c->next);
    if(debug_print == true) {
      printf("-- FreeAndMaybeCoalesce[%zu] is calling Merge1(chunk[%d], chunk[%d])\n", h, h, c->next);
      fflush(stdout);
    }
    /* Merge Function Call */
    is_AR_released_1 = Merge(h, c->next);
    if (is_AR_released_1 > 0) {
      return; // AR release했으면 리턴하자. 더이상 할 게 없다.
    }
  }
    // else{
    //   printf("[에러] region_manager_.RegionFor(c->ptr) != region_manager_.RegionFor(next_chunk->ptr)\n");
    //   exit(-1);
    // }

  // If the previous chunk is free, merge c into it and delete c.
  if (c->prev != kInvalidChunkHandle && !ChunkFromHandle(c->prev)->in_use()) {
    CHECK(region_manager_.RegionFor(c->ptr) == region_manager_.RegionFor(prev_chunk->ptr));
    coalesced_chunk = c->prev;
    RemoveFreeChunkFromBin(c->prev);
    if(debug_print == true){
      printf("-- FreeAndMaybeCoalesce[%zu] is calling Merge2(chunk[%d], chunk[%d])\n", h, c->prev, h);
      fflush(stdout);
    }
    /* Merge Function Call */
    is_AR_released_2 = Merge(c->prev, h);
    if (is_AR_released_2 > 0) {
      return;  // AR release했으면 리턴하자. 더이상 할 게 없다.
    }
  }

  /* 
    is_AR_released
      -1: Merge 자체가 콜이 된 적이 없다.
       0: Merge는 콜이 되었지만, 해당 Merge로 인해서 AR이 Release되진 않았다.
       1: Merge가 콜이 되었고, 해당 Merge로 인해서 AR이 cudaFree 되었다.
       2: Merge가 콜이 되었고, 해당 Merge로 인해서 AR이 GlobalPool로 Release 되었다.

    - is_AR_released_1 혹은 is_AR_released_2 둘 중에 하나만 0보다 큰 값을 가질 수 있다. 둘다 0보다 큰 값을 가지는 것은 말이 안된다.
      둘 다 0보다 큰 값을 가진다는 것은 똑같은 하나의 AR이 두번 Release가 되었다는 뜻이니까.

    - 둘 다 0 이면, 원래 대상이던 청크를 Bin으로 넣고 리턴.

    - 둘 다 -1 이면, 애초에 Merge 된 적도 없는 상태이다. 원래 대상이던 청크 Bin으로 넣고 리턴.
  */
  if (is_AR_released_1 == 0 || is_AR_released_2 == 0) {
  /*
    prev하고 혹은 next하고 Merge는 되었으나, AllocationRegion을 Release하진 않았다.
    이런 경우에는 "Merge된 청크"를 Bin에 넣어줘야한다.
  */
    InsertFreeChunkIntoBin(coalesced_chunk, c->sharable);
    return;
  }

  else if (is_AR_released_1 == -1 && is_AR_released_2 == -1) {
  /*
    *********************************
    *** 내가 놓치고 있던 코너 케이스! ***
    *********************************
    prev하고도 next하고도 애초에 "merge 함수 콜 자체가 되지 않았다".
    그러면 DeleteChunk함수가 콜이 안되고 그러면, AllocationRegion Free될
    기회를 애초에 가지지 못한다. 이런 경우 청크를 바로 그냥 Bin에 넣으면
    안된다.

    "애초에 해당 청크가 리전 전체를 차지하고 있을 수도 있기 때문에,
    이러한 상황에서 청크가 프리가 되면 멀지 없이 리전이 프리가 된 상황이 되는
    것이다. 

    위와 같은 경우라면, 청크의 프리가 Merge 없이 리전의 완전한 free로 이어진
    거기 때문에, 청크도 지우고, 리전도 Release를 시켜줘야한다.

    만약 아직 완전하게 Free하지 않은 리전이라면, 원래의 프리된 해당 청크 Bin에
    넣고, 리턴.
    */
    if (debug_print) {
      printf("Merge is not called for this deallocation. But, region can be freed.\n");
      fflush(stdout);
    }
    int isFreed = TryToFreeAllocationRegion(coalesced_chunk);
    if (isFreed != 0) {
      if (debug_print) {
        printf("코너 케이스!\n", Name().c_str());
        fflush(stdout);
      }
    }
    else {
      InsertFreeChunkIntoBin(coalesced_chunk, c->sharable);
    }
  }

	// It is impossible for one AR to have two times AR release for each Merge.
  else if (is_AR_released_1 > 0 && is_AR_released_2 > 0){
    printf("[에러] 한개의 AllocationRegion이 두번 Release 될 수 없는데 됐다. \n");
    fflush(stdout);
    CHECK_EQ(0,1);
  }
  else {
    printf("WHAT THE FUCK IS HAPPENING\n");
    CHECK_EQ(0,1);
  }
}

bool BFCAllocator::TracksAllocationSizes() { return true; }

size_t BFCAllocator::RequestedSize(const void* ptr) {
  mutex_lock l(lock_);
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle)
      << "Asked for requested size of pointer we never allocated: " << ptr;
  BFCAllocator::Chunk* c = ChunkFromHandle(h);
  return c->requested_size;
}

size_t BFCAllocator::AllocatedSize(const void* ptr) {
  mutex_lock l(lock_);
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle)
      << "Asked for allocated size of pointer we never allocated: " << ptr;
  BFCAllocator::Chunk* c = ChunkFromHandle(h);
  return c->size;
}

int64 BFCAllocator::AllocationId(const void* ptr) {
  mutex_lock l(lock_);
  BFCAllocator::ChunkHandle h = region_manager_.get_handle(ptr);
  CHECK(h != kInvalidChunkHandle)
      << "Asked for allocation id of pointer we never allocated: " << ptr;
  BFCAllocator::Chunk* c = ChunkFromHandle(h);
  return c->allocation_id;
}

namespace {

void RenderRegion(char* rendered, const size_t resolution,
                  const size_t total_render_size, const size_t offset,
                  const void* base_ptr, const void* ptr, const size_t size,
                  const char c) {
  const char* base_ptr_c = static_cast<const char*>(base_ptr);
  const char* ptr_c = static_cast<const char*>(ptr);

  size_t start_location =
      ((ptr_c - base_ptr_c + offset) * resolution) / total_render_size;
  CHECK_GE(start_location, 0);
  CHECK_LT(start_location, resolution);
  size_t end_location =
      ((ptr_c + size - 1 - base_ptr_c + offset) * resolution) /
      total_render_size;
  CHECK_GE(end_location, 0);
  CHECK_LT(end_location, resolution);

  for (size_t i = start_location; i <= end_location; ++i) {
    rendered[i] = c;
  }
}

}  // namespace

string BFCAllocator::RenderOccupancy() {
  // Make a buffer for the ASCII-art representation.
  const size_t resolution = 100;
  char rendered[resolution];

  // Compute the total region size to render over
  size_t total_region_size = 0;
  for (const auto& region : region_manager_.regions()) {
    total_region_size += region.memory_size();
  }

  if (total_region_size == 0) {
    return "<allocator contains no memory>";
  }

  // Start out with everything empty
  RenderRegion(rendered, resolution, total_region_size, 0, nullptr, nullptr,
               total_region_size, '_');

  size_t region_offset = 0;
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    // Then render each chunk left to right.
    while (h != kInvalidChunkHandle) {
      Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        // Render the wasted space
        size_t wasted = c->size - c->requested_size;
        if (wasted > 0) {
          RenderRegion(rendered, resolution, total_region_size,
                       region_offset + c->requested_size, region.ptr(), c->ptr,
                       wasted, 'x');
        }
        // Then the occupied space
        RenderRegion(rendered, resolution, total_region_size, region_offset,
                     region.ptr(), c->ptr, c->requested_size, '*');
      }
      h = c->next;
    }
    region_offset += region.memory_size();
  }

  return string(rendered, resolution);
}

void BFCAllocator::DumpMemoryLog(size_t num_bytes, bool print_only) {
  const std::array<BinDebugInfo, kNumBins> bin_infos = get_bin_debug_info();
  for (BinNum bin_num = 0; bin_num < kNumBins; bin_num++) {
    Bin* b = BinFromIndex(bin_num);
    const BinDebugInfo& bin_info = bin_infos[bin_num];
    CHECK_EQ(b->free_chunks.size(),
             bin_info.total_chunks_in_bin - bin_info.total_chunks_in_use);

    LOG(INFO) << "Bin (" << b->bin_size
              << "): \tTotal Chunks: " << bin_info.total_chunks_in_bin
              << ", Chunks in use: " << bin_info.total_chunks_in_use << ". "
              << strings::HumanReadableNumBytes(bin_info.total_bytes_in_bin)
              << " allocated for chunks. "
              << strings::HumanReadableNumBytes(bin_info.total_bytes_in_use)
              << " in use in bin. "
              << strings::HumanReadableNumBytes(
                     bin_info.total_requested_bytes_in_use)
              << " client-requested in use in bin.";
  }

  if(print_only == true)
    return;

  // Find the bin that we would have liked to allocate in, so we
  // can get some further analysis about fragmentation.
  Bin* b = BinForSize(num_bytes);

  LOG(INFO) << "Bin for " << strings::HumanReadableNumBytes(num_bytes)
            << " was " << strings::HumanReadableNumBytes(b->bin_size)
            << ", Chunk State: ";

  for (ChunkHandle h : b->free_chunks) {
    Chunk* c = ChunkFromHandle(h);
    LOG(INFO) << c->DebugString(this, true);
  }

  // Next show the chunks that are in use, and also summarize their
  // number by size.
  std::map<size_t, int> in_use_by_size;
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      if (c->in_use()) {
        in_use_by_size[c->size]++;
      }
      LOG(INFO) << (c->in_use() ? "Chunk" : "Free ") << " at " << c->ptr
                << " of size " << c->size;
      h = c->next;
    }
  }

  LOG(INFO) << "     Summary of in-use Chunks by size: ";
  size_t total_bytes = 0;
  for (auto& it : in_use_by_size) {
    LOG(INFO) << it.second << " Chunks of size " << it.first << " totalling "
              << strings::HumanReadableNumBytes(it.first * it.second);
    total_bytes += (it.first * it.second);
  }
  LOG(INFO) << "Sum Total of in-use chunks: "
            << strings::HumanReadableNumBytes(total_bytes);
  LOG(INFO) << "Stats: \n" << stats_.DebugString();
}

void BFCAllocator::GetStats(AllocatorStats* stats) {
  mutex_lock l(lock_);
  *stats = stats_;
}

void BFCAllocator::ClearStats() {
  mutex_lock l(lock_);
  stats_.num_allocs = 0;
  stats_.max_bytes_in_use = stats_.bytes_in_use;
  stats_.max_alloc_size = 0;
}

std::array<BFCAllocator::BinDebugInfo, BFCAllocator::kNumBins>
BFCAllocator::get_bin_debug_info() {
  std::array<BinDebugInfo, kNumBins> bin_infos;
  for (const auto& region : region_manager_.regions()) {
    ChunkHandle h = region_manager_.get_handle(region.ptr());
    while (h != kInvalidChunkHandle) {
      const Chunk* c = ChunkFromHandle(h);
      BinNum bin_num = BinNumForSize(c->size);
      BinDebugInfo& bin_info = bin_infos[bin_num];
      bin_info.total_bytes_in_bin += c->size;
      bin_info.total_chunks_in_bin++;
      if (c->in_use()) {
        bin_info.total_bytes_in_use += c->size;
        bin_info.total_requested_bytes_in_use += c->requested_size;
        bin_info.total_chunks_in_use++;
      } else {
        Bin* bin = BinFromIndex(bin_num);
        CHECK_EQ(bin->free_chunks.count(h), 1);
        CHECK_EQ(c->bin_num, bin_num);
      }
      h = c->next;
    }
  }
  return bin_infos;
}

}  // namespace tensorflow
