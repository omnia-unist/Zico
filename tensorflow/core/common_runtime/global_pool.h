#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GLOBAL_POOL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GLOBAL_POOL_H_

#include <iostream>
#include "stdio.h"
#include <vector>
#include <utility>
#include <ctime>
#include <chrono>
#include <map>
#include <queue>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include "tensorflow/core/common_runtime/gangmuk_parser.h"
#include <fstream>

namespace tensorflow {

class cr_GlobalPool;

typedef size_t ChunkHandle;

struct Region {
	size_t region_id = -1; // last region id for this AR
	size_t size = -1;
	bool sharable = false; // AR's last dedicated job id
  std::string allocator_name = "not initialized";
	void* memory_ptr = nullptr; // addr of this AR memory space
  // std::unique_ptr<ChunkHandle[]> handles;
  ChunkHandle* handles = nullptr;
  unsigned long long alloc_time = 0;
};

struct RegionEvent {
  int event_type;
  bool is_forward;
  int idx;
  unsigned long long event_time;
  size_t region_size;
  size_t total_using_memory;
  RegionEvent(int event_type_, bool is_forward_, int idx_, unsigned long long event_time_,
              size_t region_size_, size_t total_using_memory_)
      : event_type(event_type_),
        idx(idx_),
        is_forward(is_forward_),
        event_time(event_time_),
        region_size(region_size_),
        total_using_memory(total_using_memory_){};
};

struct Chunk_MetaData {
  std::string caller;
  std::string kernel_name;
  bool set_output;
  bool sharable;
  bool is_gradient;
  bool having_bridge;
  unsigned long long birth_time;
  unsigned long long lifespan;
  bool is_backward;
  long long allocation_id;
  long long gangmuk_allocation_id;
  size_t chunk_size;
  // int executorstate_id;
  int executorimpl_id;
};

class BFCStatus {
  public: // member variable
    cr_GlobalPool* cr_global_pool_;
    std::string name;
    int id;
    volatile size_t chunk_bytes = 0;
    volatile size_t region_bytes = 0;
 
    std::string forward_start_node_name = "not_initialized";
    std::string backward_start_node_name = "not_initialized";
 
    long long temp_chunk_peak_allocation_id = 0;
    long long chunk_peak_allocation_id = 0;
    long long temp_chunk_peak_bytes = 0;
    volatile long long chunk_peak_bytes = 0;
    volatile unsigned long long temp_chunk_peak_time = 0;
    volatile unsigned long long chunk_peak_time = 0;
 
    long long temp_region_peak_allocation_id = 0;
    size_t temp_region_peak_bytes = 0;
    unsigned long long temp_region_peak_time = 0;
    volatile long long region_peak_allocation_id = 0;
    volatile size_t region_peak_bytes = 0;
    volatile unsigned long long region_peak_time = 0;

    long long temp_backward_region_peak_allocation_id = 0;
    size_t temp_backward_region_peak_bytes = 0;
    unsigned long long temp_backward_region_peak_time = 0;
    volatile long long backward_region_peak_allocation_id = 0;
    volatile size_t backward_region_peak_bytes = 0;
    volatile unsigned long long backward_region_peak_time = 0;
 
    int current_step_id = 0;
    volatile long long curr_united_id = 0;
    volatile long long last_united_id = 0;
    volatile long long allocation_id = 0;
    volatile long long deallocation_id = 0;
    volatile long long last_allocation_id = 0;
    volatile long long last_deallocation_id = 0;
    volatile bool reach_to_last_alloc = false;
    volatile bool reach_to_last_dealloc = false;
    volatile bool reach_to_peak_alloc = false;
    
    unsigned long long start_time = 0;
    unsigned long long actual_start_time = 0;
    unsigned long long backward_start_time = 0;
    unsigned long long end_time = 0;

    volatile unsigned long long iter_time = 0;
    volatile unsigned long long forward_time = 0;
    volatile unsigned long long backward_time = 0;

    unsigned long long total_timeshift = 0;
    unsigned long long num_timeshift = 0;
    // unsigned long long timeshift_unit = 500; // GlobalPool 클래스로 이동

    volatile bool is_forward = false;
    volatile bool is_backward = false;
    volatile bool almost_end = false;
 

    // 이 delay들에 문제가 있다.
    bool cuda_alloc_is_delayed = false;
    unsigned long long cuda_alloc_delay_start = 0;
    unsigned long long cuda_alloc_delay_end = 0;
    unsigned long long cuda_alloc_delay_time = 0;
 
    int region_event_idx = 0;
    std::vector<RegionEvent> prev_region_events;
    std::vector<RegionEvent> curr_region_events;
 
    bool first_isolated_profiling = true;
    bool first_isolated_profiling_2 = true;
    unsigned long long isolated_profiling_start_time = 0;
    unsigned long long isolated_profiling_start_time_2 = 0;

    unsigned long long isolated_last_ms = 0;
    unsigned long long isolated_last_ms_2 = 0;
    size_t isolated_last_memory = 0;
    size_t isolated_last_memory_2 = 0;
    long long isolated_profiling_id = 0;
    long long isolated_profiling_id_2 = 0;
    std::vector<std::pair<long long, size_t>> isolated_profiling_info;
    std::vector<std::pair<long long, size_t>> isolated_profiling_info_2;
    
 
    unsigned long long colocated_last_ms = 0;
    size_t colocated_last_memory = 0;
    std::vector<std::pair<long long, size_t>> prev_prev_colocated_profiling_info;
    std::vector<std::pair<long long, size_t>> prev_colocated_profiling_info;
    std::vector<std::pair<long long, size_t>> curr_colocated_profiling_info;

    bool finish_isolated_profiling = false;
 
    volatile int num_region_events_setup = 0;
    volatile bool region_events_setup_flag = false;
    int num_region_events_init_steps = 2;
 
    int start_aligning_step;
    int num_isolated_profiling;
    long long colocated_profiling_id = 0;
 
    size_t bytes_to_mb = 1048576;
 
    int incrementer = 0;

    std::vector<std::tuple<void*, int, uint64, uint64, uint64>> region_lifetime;
    // std::vector<std::pair<void*, unsigned long long>> region_lifetime;

    // std::vector<std::tuple<std::string, int, std::string, int, int, int64, size_t, int64, int64, int64, size_t, uint64, int, int, int, int, uint64, int>> bfclog_vec;
    std::vector<std::tuple<std::string, int, int, int, int64, size_t, int64, int64, int64, size_t, int64, uint64, bool>> bfclog_vec;
    

    long long aggregated_chunk_bytes;
    long long aggregated_region_bytes;
    void PushBackBFCLog(int step_id, int alloc_dealloc, Chunk_MetaData chunk_metadata, long long bytes_in_use,  size_t total_region_allocated_bytes_, long long global_pool_holding_bytes);
    void BFCLogFileWrite();

    mutable mutex lock_bfcstatus_;
 
  public:  // member method
    BFCStatus();
 
    BFCStatus(int bfc_id, std::string bfc_name);
 
    BFCStatus(int bfc_id, std::string bfc_name, int start_aligning_step, int num_isolated_profiling, cr_GlobalPool* cr_global_pool_);
 
    void InitBFCStatus();
 
    bool NewStepUpdate(int step_id);
 
    void AllocationUpdate(long long bytes_in_use, int step_id, size_t total_region_allocated_bytes_);

    void UpdateChunkPeakBytes(long long bytes_in_use);

    void UpdateRegionPeakBytes(size_t total_region_allocated_bytes_);

    void UpdateBackwardRegionPeakBytes(size_t total_region_allocated_bytes_);

    bool CheckIfBackward(int step_id);
    bool CheckIfAlmostEnd(int step_id);

    bool IsLastAlloc() {
      if (allocation_id != 0 && allocation_id == last_allocation_id) {
        // printf("ReachLastAllocation,%s,step_id,%d,last_allocation_id,%lld\n", name.c_str(), current_step_id, last_allocation_id);
        return true;
      }
      return false;
    }
    bool IsLastDealloc() {
      if (deallocation_id != 0 && deallocation_id == last_deallocation_id) {
        // printf("ReachLastDeallocation,%s,step_id,%d,last_deallocation_id,%lld\n", name.c_str(), current_step_id, deallocation_id);
        return true;
      }
      return false;
    }
 
    void DeallocationUpdate(long long bytes_in_use, int step_id, size_t total_region_allocated_bytes_);
 
    void RecordRegionFreeEvent(size_t region_size);
    void RecordRegionAllocEvent(size_t region_size);
    void IsolatedProfiling(size_t total_region_allocated_bytes_, int alloc_or_dealloc);
    void IsolatedProfiling_2(size_t total_region_allocated_bytes_, int alloc_or_dealloc);
    void ColocatedProfiling(size_t total_region_allocated_bytes_, int alloc_or_dealloc);

    void FillUpIsolatedProfiling() {
      for (int i = 0; i < iter_time/1000 - isolated_last_ms - 1; i++) {
        isolated_profiling_info.push_back(std::make_pair(curr_united_id, isolated_last_memory));
        isolated_profiling_id++;
        printf("IsolatedProfiling,%s(step%d),%s,current_ms,%llu,isolated_last_ms,%llu,idx,%lld,region_bytes,%zu\n",
                name.c_str(),current_step_id-1, "lastfillup", 
                iter_time/1000, isolated_last_ms,
                isolated_profiling_id,
                isolated_last_memory/bytes_to_mb);
      }
    }

    void FillUpIsolatedProfiling_2() {
      for (int i = 0; i < iter_time/1000 - isolated_last_ms_2 - 1; i++) {
        isolated_profiling_info_2.push_back(std::make_pair(curr_united_id, isolated_last_memory_2));
        isolated_profiling_id_2++;
        printf("IsolatedProfiling_2,%s(step%d),%s,current_ms,%llu,isolated_last_ms_2,%llu,idx,%lld,region_bytes,%zu\n",
                name.c_str(),current_step_id-1, "lastfillup", 
                iter_time/1000, isolated_last_ms_2,
                isolated_profiling_id_2,
                isolated_last_memory_2 / bytes_to_mb);
      }
    }

    void FillUpColocatedProfiling() {
      for (int i = 0; i < iter_time/1000 - colocated_last_ms - 1; i++) {
        curr_colocated_profiling_info.push_back(std::make_pair(curr_united_id, colocated_last_memory));
        colocated_profiling_id++;
        printf("ColocatedProfiling,%s,%d,%s,current_ms,%llu,colocated_last_ms,%llu,idx,%lld,region_bytes,%zu\n",
                name.c_str(),current_step_id-1, "lastfillup", 
                iter_time/1000, colocated_last_ms,
                colocated_profiling_id,
                colocated_last_memory/bytes_to_mb);
      }
    }

    void PrintIsolatedProfilingResult() {
      for (int i = 0; i < isolated_profiling_info.size(); i++) {
        printf("PrintIsolatedProfilingResult,%s,%d,idx,%d,curr_united_id,%lld,Memory,%zu\n",
                name.c_str(), current_step_id-1, i, isolated_profiling_info[i].first, isolated_profiling_info[i].second/bytes_to_mb);
        fflush(stdout);
      }
    }

    void PrintIsolatedProfilingResult_2() {
      for (int i = 0; i < isolated_profiling_info_2.size(); i++) {
        printf("PrintIsolatedProfilingResult,%s,%d,idx,%d,curr_united_id,%lld,Memory,%zu\n",
                name.c_str(), current_step_id-1, i, isolated_profiling_info_2[i].first, isolated_profiling_info_2[i].second/bytes_to_mb);
        fflush(stdout);
      }
    }

    void PrintColocatedProfilingResult() {
      for (int i = 0; i < curr_colocated_profiling_info.size(); i++) {
        printf("PrintColocatedProfilingResult,%s,%d,idx,%d,curr_united_id,%lld,Memory,%zu\n",
                name.c_str(), current_step_id-1,i, curr_colocated_profiling_info[i].first, curr_colocated_profiling_info[i].second/bytes_to_mb);
        fflush(stdout);
      }
    }
 
    bool IsNewGlobalStep(int step_id) {
      return step_id != current_step_id;
      // return step_id > current_step_id;
    }
    
    void SendIsolatedProfilingFinishSignal() {
      finish_isolated_profiling = true;
      printf("DAMN,FinishIsolatedProfiling,%d,%s\n", current_step_id-1, name.c_str());
    }

    bool CondForIsolatedRun() {
      if (current_step_id >= start_aligning_step && current_step_id <= start_aligning_step + num_isolated_profiling - 1) {
        return true;
      }
      return false;
    }

    bool CondForIsolatedRun_2() {
      if (current_step_id >= start_aligning_step && current_step_id <= start_aligning_step + num_isolated_profiling - 1) {
        return true;
      }
      return false;
    }

    bool CondForIsolatedProfiling() {
      // if (current_step_id == start_aligning_step + (num_isolated_profiling - 1) { // 15 + 10 - 1 = 24
      if (current_step_id == start_aligning_step + (num_isolated_profiling - 2)) { // 15 + 10 - 1 = 23
        return true;
      }
      return false;
    }

    bool CondForIsolatedProfiling_2() {
      if (current_step_id == start_aligning_step + num_isolated_profiling + 4) { // 15 + 10 + 4 = 29
        return true;
      }
      return false;
    }


    bool CondForColocatedProfiling() {
      if (current_step_id >= start_aligning_step + num_isolated_profiling - 1) {
        return true;
      }
      return false;
    }
    void PrintLifeTimeVector(int step_id);
    void EndOfOldStep(int step_id);
    void InitAllocationId();
    void SetChunkPeakMemory();
    void SetRegionPeakMemory();
    void SetBackwardRegionPeakMemory();
    void UpdateIterationTime();
    void ActualStartOfNewStep();


    int GetCurrIsolatedProfilingIdx() {
      for (int curr_idx = 0; curr_idx < isolated_profiling_info.size(); ++curr_idx) {
        if (isolated_profiling_info[curr_idx].first >= curr_united_id) {
          return curr_idx;
        }
      }
    }

    int GetCurrIsolatedProfilingIdx_2() {
      for (int curr_idx = 0; curr_idx < isolated_profiling_info_2.size(); ++curr_idx) {
        if (isolated_profiling_info_2[curr_idx].first >= curr_united_id) {
          return curr_idx;
        }
      }
    }

}; // class BFCStatus

class cr_GlobalPool {
	public:
    mutable mutex lock_;
    // mutable mutex lock_recordmemop_;

    bool initialized = false;
    GM_Parser parser;
    std::string name = "cr_GlobalPool";
		bool print_globalpool = false;
		std::vector<Region> free_regions;
    int num_virtual_gpu = 0;
    int start_aligning_step = 0;
    bool global_pool_release_ar;
    bool global_pool_release_all;
    bool cuda_alloc_test = false;
    int num_allocators = 0;
    unsigned long long base_micro = 0;

    std::map<int, BFCStatus> bfc_status_vec;

    int num_bookkeepers = 0;
    int next_bookkeeper_id = 0;

    int print_bfc;
    int print_summary_step; // not being used anymore. It is replaced with allocation_info_registration_step+1
    int allocation_info_registration_step;
    int print_bfc_step;

    long long global_pool_holding_bytes = 0;

    volatile long long aggregated_chunk_bytes = 0;
    volatile long long aggregated_region_bytes = 0;
    
    int tight_fit = 0;
    std::string coordinator_policy = "not_init";
    bool turn_on_coordinator = false;
    int print_period = 1;
    bool use_colocated_simulator = false;

    // tensorflow system memory는 이미 빠져있음. (gpu_device.cc 참조)
    long long gpu_memory_limit = 0; // 말 그대로 gpu_memory_limit, 이 넘버가 GPU안에 fit시키는 hard deadline (deadline은 아니고 뭐라고 해야되냐..)
    volatile long long buffer_size = 0; // gpu_memory_limit * buffer_ratio
    volatile long long memory_budget = 0; // gpu_memory_limit - buffer_size
    volatile long long total_aggregated_alloc_memory = 0; // 모든 BFC에 할당된 메모리 합친거
    volatile long long remaining_memory_budget = 0; // budget에서 각 bfc에 할당되고 남은 메모리양
    volatile long long available_gpu_memory = 0; // = gpu_memory_limit - total_aggregated_alloc_memory
                                              // = remaining_memory_budget + buffer_size

    volatile long long touched_host_memory = 0;
    volatile bool host_memory_touch = false;
    volatile bool buffer_touch = false;
    volatile long long buffer_ratio = 50; // gpu_memory_limit 중에 buffer를 몇퍼센트로 할지 맨처음에 셋업하는 값
    long long buffer_control_unit = 5; // host memory를 터치할 때 그리고 budget을 넘어서서 buffer를 터치할 때 buffer를 증가시키는 단위 (퍼센티지임.)

    int num_isolated_profiling = 10;

    size_t bytes_to_mb = 1048576;

    unsigned long long timeshift_unit = 500;

  public:
    cr_GlobalPool();
    ~cr_GlobalPool();

    void InitGlobalPool(unsigned long long base_micro, size_t memory_limit_);

    bool ModelBasedCoordination() {
      if((coordinator_policy == "ModelBased" || coordinator_policy == "Spatial" || coordinator_policy == "TickTock") && num_virtual_gpu >= 2 && turn_on_coordinator == false) {
        return true;
      }
      return false;
    }

    void PrintRegionMove(int step_id, std::string action, std::string src, std::string dst,
                         Region region, size_t total_region_alloc_bytes,
                         unsigned long long elapsed, bool sharable_,
                         bool is_forward);
                         
    // bool FreeRegionToGlobalPool(void* memory_ptr, size_t free_region_size, size_t region_id,
    //                    bool sharable, const std::string allocator_name,
    //                    int current_step_id, unsigned long long elapsed, int bfc_id, 
    //                    size_t total_region_allocated_bytes_);
    bool FreeRegionToGlobalPool(void* memory_ptr, size_t free_region_size, size_t region_id,
                       bool sharable, const std::string allocator_name,
                       int current_step_id, unsigned long long elapsed, int bfc_id, 
                       size_t total_region_allocated_bytes_, ChunkHandle* handles_,
                       unsigned long long birth_time);

    Region AllocRegionFromGlobalPool(size_t rounded_bytes, size_t minimum_required_region_size,
                                     const std::string dst_allocator_name, int current_step_id,
                                     unsigned long long elapsed,
                                     bool sharable,
                                     size_t total_region_allocated_bytes_, int bfc_id);

    bool TryCudaAlloc(size_t minimum_required_region_size, const std::string dst_allocator_name, int bfc_id, int step_id) {
      if (!cuda_alloc_test) {
        return true;
      }
      if (available_gpu_memory >= minimum_required_region_size) {
        return true;
      }
      else {
        printf("CudaAllocPossible,%s,Fail,available_gpu_memory(%lldMB) < "
               "minimum_required_region_size(%zuMB), gpu_memory_limit(%lld)MB, BFC_0(%zuMB,is_forward[%d]), "
               "BFC_1(%zuMB,is_forward[%d])\n",
               dst_allocator_name.c_str(), 
               available_gpu_memory / bytes_to_mb,
               minimum_required_region_size / bytes_to_mb, 
               gpu_memory_limit / bytes_to_mb,
               bfc_status_vec[0].region_bytes / bytes_to_mb,
               ForwardToString(bfc_status_vec[0].is_forward),
               bfc_status_vec[1].region_bytes / bytes_to_mb,
               ForwardToString(bfc_status_vec[1].is_forward));
        return false;
      }
    }

    int NumRegion() { return free_regions.size(); }

    size_t UpdateAggChunkBytes() {
      aggregated_chunk_bytes = 0;
      for (int i = 0; i < bfc_status_vec.size(); i++) {
        aggregated_chunk_bytes += bfc_status_vec[i].chunk_bytes;
      }
      for (int i = 0; i < bfc_status_vec.size(); i++) {
        bfc_status_vec[i].aggregated_chunk_bytes = aggregated_chunk_bytes;
      }
    }

    size_t UpdateAggRegionBytes() {
      aggregated_region_bytes = 0;
      for (int i = 0; i < bfc_status_vec.size(); i++) {
        aggregated_region_bytes += bfc_status_vec[i].region_bytes;
      }
      for (int i = 0; i < bfc_status_vec.size(); i++) {
        bfc_status_vec[i].aggregated_region_bytes = aggregated_region_bytes;
      }
    }

    void PrintSimulateResult(std::string simulator_name, std::string result, int step_id, BFCStatus& me, BFCStatus& other, size_t me_estimation, size_t other_estimation) {
      printf("%s,%s,%s,step,%d,%s[%s,%zu]+%s[%s,%zu],memory_budget[%zu]\n",
              simulator_name.c_str(),
              result.c_str(),
              me.name.c_str(),
              step_id,
              me.name.c_str(),
              ForwardToString(me.is_forward).c_str(),
              me_estimation/bytes_to_mb,
              other.name.c_str(),
              ForwardToString(other.is_forward).c_str(),
              other_estimation/bytes_to_mb,
              memory_budget/bytes_to_mb);
    }

    void AddNewBookkeeper();

    void AddNewBFCStatus(int bfc_id, std::string bfc_name);

    void ResetMemoryComposition(long long buffer_ratio, int step_id) {
      long long prev_buf_size = buffer_size;
      long long prev_memory_budget = memory_budget;
      buffer_size = (gpu_memory_limit * buffer_ratio) / 100;
      memory_budget = gpu_memory_limit - buffer_size;
      printf("ResetMemoryComposition,buffer_size,step_id,%d,from,%lld,MB,to,%lld,MB\n", step_id, prev_buf_size/bytes_to_mb, buffer_size/bytes_to_mb);
      printf("ResetMemoryComposition,memory_budget,step_id,%d,from,%lld,MB,to,%lld,MB\n",step_id, prev_memory_budget/bytes_to_mb, memory_budget/bytes_to_mb);
    }


    void UpdateRemainingBudget(std::string caller);

    void UpdateAvailableGPUMemory(std::string caller);

    void RecordCUDA(std::string name, int step_id, size_t bytes, bool sharable_, bool is_backward, std::string mem_op);
    void RecordCUDAAlloc(std::string name, int step_id, size_t bytes, bool sharable_, bool is_backward);
    void RecordCUDAFree(std::string name, int step_id, size_t bytes, bool sharable_, bool is_backward);

    bool CudaAllocPossible(std::string allocator_name, size_t rounded_bytes, size_t minimum_required_region_size);

    void SimulateIfNewStep(int bfc_id, int step_id);

    void CoordinatorSimulateIfNewStep(int bfc_id, int step_id);

    bool IsolatedSimulator(int bfc_id, int step_id);
    bool IsolatedSimulator_2(int bfc_id, int step_id);
    
    bool CondForIsolatedSimulator(int step_id) {
      if ((step_id > start_aligning_step + num_isolated_profiling) && 
          (step_id <= start_aligning_step + num_isolated_profiling + 5)) { // 26 ~ 30
        return true;
      }
      return false;
    }
    bool CondForIsolatedSimulator_2(int step_id) {
      if (step_id > start_aligning_step + num_isolated_profiling + 5) { // 31 ~
        return true;
      }
      return false;
    }

    bool SpatialSimulator(int bfc_id, int step_id);

    bool TickTockScheduler(int bfc_id, int step_id);
    
    bool ColocatedSimulator(int bfc_id, int step_id);

    bool Simulate(int bfc_id, int step_id);
    
    bool IfHostMemoryTouched();

    void IncreaseBufferSize(int step_id);

    void PrintBFCLog(int step_id, int bfc_id, std::string alloc_or_dealloc, int64 allocation_id, 
                     size_t chunk_size, bool is_gradient, bool sharable, bool set_output, unsigned long long lifespan, 
                     long long bytes_in_use, size_t total_region_allocated_bytes_, 
                     std::string which_op_allocate_function, std::string op_kernel_name);


    // void PrintBFCLog_2(int step_id, int bfc_id, std::string alloc_or_dealloc, Chunk_MetaData chunk_metadata, long long bytes_in_use, size_t total_region_allocated_bytes_);

    void RecordMemOperation(int step_id,
                            int bfc_id,
                            int alloc_dealloc,
                            Chunk_MetaData chunk_metadata,
                            long long bytes_in_use,
                            size_t total_region_allocated_bytes_);

    /*
      GPU_1_bfc, step 25
      첫번째 파트 메뉴얼 얼라이닝.
    */
    void FirstAligning(int step_id);

    void BFC1_Sleep_Until_BFC0_Finishes_IsolatedProfiling(int step_id);

    void BFC0_Sleep_Until_BFC1_Finishes_IsolatedProfiling(int step_id);

    Region NullRegion();

    std::string SharableToString(bool sharable) {
      if (sharable) {
        return "sharable";
      }
      return "non_sharable";
    }

    std::string ForwardToString(bool sharable) {
      if (sharable) {
        return "forward";
      }
      return "backward";
    }
}; 

}  // namespace tensorflow

#endif
