#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/common_runtime/global_pool.h"
#include <string>

namespace tensorflow {

BFCStatus::BFCStatus() {
  id = -1;
  name = "noname_yet";
}

BFCStatus::BFCStatus(int bfc_id, std::string bfc_name, int start_aligning_step_, int num_isolated_profiling_, cr_GlobalPool* cr_global_pool) {
  id = bfc_id;
  name = bfc_name;
  cr_global_pool_ = cr_global_pool;
  start_aligning_step = start_aligning_step_;
  num_isolated_profiling = num_isolated_profiling_;
  // bfclog_vec.reserve(1000000);
}

void BFCStatus::PrintLifeTimeVector(int step_id) {
  for (auto it = region_lifetime.begin(); it != region_lifetime.end(); ++it) {
    printf("RegionLifetime,GPU_%d_bfc,%d,memory_ptr,%x,region_id,%d,birthtime,%llu,deathtime,%llu,lifetime,%llu\n", id, step_id, std::get<0>(*it), std::get<1>(*it), std::get<2>(*it), std::get<3>(*it), std::get<4>(*it));
  }
  // for (int i = 0; i < region_lifetime.size(); ++i) {
  //   printf("RegionLifetime,GPU_%d_bfc,%d,memory_ptr,%x,lifetime,%llu\n", id, step_id, region_lifetime[i].first, region_lifetime[i].second);
  // }
  region_lifetime.clear();
}

 void BFCStatus::PushBackBFCLog(int step_id, int alloc_dealloc, Chunk_MetaData chunk_metadata, long long bytes_in_use,  size_t total_region_allocated_bytes_, long long global_pool_holding_bytes) {
  // std::tuple<std::string, int, std::string, int, int, int64, size_t, int64, int64, int64, size_t, uint64, int, int, int, int, uint64,int> tup = std::make_tuple("BFCRecord_5", 
  //                   step_id, 
  //                   "GPU_"+std::to_string(id)+"_bfc", 
  //                   alloc_dealloc, 
  //                   chunk_metadata.executorimpl_id,
  //                   chunk_metadata.gangmuk_allocation_id,
  //                   chunk_metadata.chunk_size,
  //                   bytes_in_use,
  //                   aggregated_chunk_bytes,
  //                   total_region_allocated_bytes_,
  //                   aggregated_region_bytes,
  //                   Env::Default()->NowNanos(),
  //                   chunk_metadata.is_gradient,
  //                   chunk_metadata.sharable,
  //                   chunk_metadata.set_output,
  //                   chunk_metadata.is_backward,
  //                   chunk_metadata.lifespan/1000,
  //                   chunk_metadata.having_bridge);
  std::tuple<std::string, int, int, int, int64, size_t, int64, int64, int64, int64, int64, uint64, bool> tup = std::make_tuple("BFCRecord_5", 
                                                                                                    step_id, 
                                                                                                    id, 
                                                                                                    alloc_dealloc, 

                                                                                                    chunk_metadata.gangmuk_allocation_id,
                                                                                                    chunk_metadata.chunk_size,

                                                                                                    bytes_in_use,
                                                                                                    aggregated_chunk_bytes,
                                                                                                    total_region_allocated_bytes_,
                                                                                                    aggregated_region_bytes,
                                                                                                    global_pool_holding_bytes,
                                                                                                    Env::Default()->NowNanos()/1000,
                                                                                                    chunk_metadata.sharable
                                                                                                    );
  bfclog_vec.push_back(tup);
}

void BFCStatus::BFCLogFileWrite() {
    std::ofstream fout;
    // std::string filename = "/home/gangmuk/mnt/ssd2/V100/tf_bfc_log_BFC" + std::to_string(id) + ".output";
    std::string filename = "/home/gangmuk/mnt/ssd2/V100/tf_bfc_log.output";
    fout.open(filename, std::ofstream::out | std::ofstream::app);
    for (auto it = bfclog_vec.begin(); it != bfclog_vec.end(); ++it) {
      // fout << std::get<0>(*it) << "," 
      // << std::get<1>(*it) << "," 
      // << std::get<2>(*it) << "," 
      // << std::get<3>(*it) << "," 
      // << std::get<4>(*it) << "," 
      // << std::get<5>(*it) << "," 
      // << std::get<6>(*it) << "," 
      // << std::get<7>(*it) << "," 
      // << std::get<8>(*it) << "," 
      // << std::get<9>(*it) << "," 
      // << std::get<10>(*it) << "," 
      // << std::get<11>(*it) << "," 
      // << std::get<12>(*it) << "," 
      // << std::get<13>(*it) << "," 
      // << std::get<14>(*it) << "," 
      // << std::get<15>(*it) << "," 
      // << std::get<16>(*it) << "," 
      // << std::get<17>(*it) << "\n";

      fout << std::get<0>(*it) << "," 
           << std::get<1>(*it) << "," 
           << std::get<2>(*it) << "," 
           << std::get<3>(*it) << "," 
           << std::get<4>(*it) << "," 
           << std::get<5>(*it) << "," 
           << std::get<6>(*it) << "," 
           << std::get<7>(*it) << "," 
           << std::get<8>(*it) << "," 
           << std::get<9>(*it) << "," 
           << std::get<10>(*it) << "," 
           << std::get<11>(*it) << "," 
           << std::get<12>(*it) << "\n";
    }
    fout.close();
    size_t vectorsize_mb =  (bfclog_vec.size() * sizeof(bfclog_vec)) / (1024*1024);
    // printf("Accumulated bfclog_vec size, %zuMB\n", vectorsize_mb);
    bfclog_vec.clear();
}

// BFCAllocator에서 직접 콜한다.
void BFCStatus::EndOfOldStep(int step_id) {
  UpdateIterationTime();
  current_step_id = step_id;
  is_forward = false;
  is_backward = false;
  almost_end = false;
  InitAllocationId();
  SetChunkPeakMemory();
  SetRegionPeakMemory();
  SetBackwardRegionPeakMemory();
  isolated_last_ms = 0;
  isolated_last_ms_2 = 0;
  isolated_last_memory = 0;
  isolated_last_memory_2 = 0;
  isolated_profiling_id = 0;
  isolated_profiling_id_2 = 0;
  num_timeshift = 0;
  // PrintLifeTimeVector(step_id);
  /*
    GPU_0_bfc, GPU_1_bfc 둘다, step 25 Isolated Profiling이 끝났다고 표시.
    서로의 Isolated Profiling을 기다려주는 걸 푸는 부분.
  */
  if (step_id == start_aligning_step + num_isolated_profiling) {
  //   FillUpIsolatedProfiling();
  //   PrintIsolatedProfilingResult();
    SendIsolatedProfilingFinishSignal();
  }
  colocated_last_ms = 0;
  colocated_last_memory = 0;
  colocated_profiling_id = 0;
  if (step_id >= start_aligning_step + num_isolated_profiling) {
    // FillUpColocatedProfiling();
    // PrintColocatedProfilingResult();
    prev_prev_colocated_profiling_info = prev_colocated_profiling_info; // copy
    prev_colocated_profiling_info = curr_colocated_profiling_info; // copy
    curr_colocated_profiling_info = std::vector<std::pair<long long, size_t>>();
  }
}

// [SimulateIfNewStep 함수의 구현 구조]
//    EndOfOldStep
//     - BFC1_Sleep_Until_BFC0_Finishes_IsolatedProfiling
//     - BFC0_Sleep_Until_BFC1_Finishes_IsolatedProfiling
//     - FirstAligning
//     - Simulate
//    ActualStartOfNewStep
// ActualStartOfNewStep를 실행하기 전까지는 진짜 새로운 스텝이 시작됐다고 볼 수 없다.
// 가운데에 있는 4개의 함수에서 waiting을 시키기 때문이다. (가장 메인인 부분으로 Simulate함수)
// 간단히 말해 "step이 마무리 되었음에도 새로운 step이 시작된건 아니다."
// 새로운 스텝이 시작되었다는 것을 의미하는 부분은 EndOfOldStep에서 업데이트가 되어선 안된다!
void BFCStatus::ActualStartOfNewStep() {
  /* 실질적인 forward의 시작 */
  is_forward = true;
  actual_start_time = Env::Default()->NowMicros();
  cuda_alloc_delay_start = 0;
  cuda_alloc_delay_end = 0;
  cuda_alloc_delay_time = 0;
  cuda_alloc_is_delayed = false;
}

void BFCStatus::UpdateIterationTime() {
  forward_time = backward_start_time - actual_start_time;
  backward_time = Env::Default()->NowMicros() - backward_start_time;
  end_time = Env::Default()->NowMicros();
  iter_time = end_time - start_time;
  start_time = end_time;
  printf("BFCAllocatorView,%s,EndOfIteration,step,%d,TotalIterationTime,%llu,ActualIterationTime,%llu,TimeShift,%llu,"
        "PeakChunkMemory,%lld,PeakRegionMemory,%lld,ChunkPeakAllocationId,%llu,RegionPeakAllocationId,%llu\n",
        name.c_str(),
        current_step_id,
        iter_time / 1000,
        (iter_time - total_timeshift) / 1000,
        total_timeshift / 1000,
        chunk_peak_bytes / bytes_to_mb,
        region_peak_bytes / bytes_to_mb,
        chunk_peak_allocation_id,
        region_peak_allocation_id);
  // fflush(stdout);
}

void BFCStatus::InitAllocationId() {
  last_united_id = std::max(last_united_id, curr_united_id);
  last_allocation_id = allocation_id;
  last_deallocation_id = deallocation_id;
  curr_united_id = 0;
  allocation_id = 0;
  deallocation_id = 0;
}

void BFCStatus::SetChunkPeakMemory() {
  chunk_peak_allocation_id = temp_chunk_peak_allocation_id;
  chunk_peak_bytes = temp_chunk_peak_bytes;
  chunk_peak_time = temp_chunk_peak_time;
  temp_chunk_peak_allocation_id = 0;
  temp_chunk_peak_bytes = 0;
  temp_chunk_peak_time = 0;
}
void BFCStatus::SetRegionPeakMemory() {
  region_peak_allocation_id = temp_region_peak_allocation_id;
  region_peak_bytes = temp_region_peak_bytes;
  region_peak_time = temp_region_peak_time;
  temp_region_peak_allocation_id = 0;
  temp_region_peak_bytes = 0;
  temp_region_peak_time = 0;
}
void BFCStatus::SetBackwardRegionPeakMemory() {
  backward_region_peak_allocation_id = temp_backward_region_peak_allocation_id;
  backward_region_peak_bytes = temp_backward_region_peak_bytes;
  backward_region_peak_time = temp_backward_region_peak_time;
  temp_backward_region_peak_allocation_id = 0;
  temp_backward_region_peak_bytes = 0;
  temp_backward_region_peak_time = 0;
}
////////////////////////////////////////////////////////////////////////////////////
void BFCStatus::UpdateChunkPeakBytes(long long bytes_in_use) {
  if (bytes_in_use > temp_chunk_peak_bytes) {
    temp_chunk_peak_bytes = bytes_in_use;
    temp_chunk_peak_allocation_id = allocation_id;
    temp_chunk_peak_time = Env::Default()->NowMicros() - start_time;
  }
}

void BFCStatus::UpdateRegionPeakBytes(size_t total_region_allocated_bytes_) {
  if (total_region_allocated_bytes_ > temp_region_peak_bytes) {
    temp_region_peak_bytes = total_region_allocated_bytes_;
    temp_region_peak_allocation_id = allocation_id;
    temp_region_peak_time = Env::Default()->NowMicros() - start_time;
  }
}

void BFCStatus::UpdateBackwardRegionPeakBytes(size_t total_region_allocated_bytes_) {
  if (total_region_allocated_bytes_ > temp_backward_region_peak_bytes) {
    temp_backward_region_peak_bytes = total_region_allocated_bytes_;
    temp_backward_region_peak_allocation_id = allocation_id;
    temp_backward_region_peak_time = Env::Default()->NowMicros() - start_time;
  }
}

bool BFCStatus::CheckIfBackward(int step_id) {
  if (allocation_id == region_peak_allocation_id) {
    is_forward = false;
    is_backward = true;
    backward_start_time = Env::Default()->NowMicros();
    // printf("%s,%d,BackwardStart,RegionPeakAllcationId,%lld,RegionPeakMemory,%zu\n",
    //         name.c_str(), step_id, allocation_id, region_peak_bytes / bytes_to_mb);
    return true;
  }
  return false;
}

bool BFCStatus::CheckIfAlmostEnd(int step_id) {
  float percentile = 0.99;
  float converted_last_allocation_id = (float)last_allocation_id;
  float converted_allocation_id = (float)allocation_id;
  if (almost_end == false && converted_allocation_id >= converted_last_allocation_id * percentile) {
    almost_end = true;
    // printf("%s,%d,almost_end,%lld >= %lld*%f\n", name.c_str(), step_id, allocation_id, last_allocation_id, percentile);
    return true;
  }
  return false;
}

void BFCStatus::AllocationUpdate(long long bytes_in_use, int step_id, size_t total_region_allocated_bytes_) {
  ++curr_united_id;
  ++allocation_id;
  chunk_bytes = bytes_in_use;
  region_bytes = total_region_allocated_bytes_;
  UpdateChunkPeakBytes(bytes_in_use);
  UpdateRegionPeakBytes(total_region_allocated_bytes_);
  CheckIfBackward(step_id);
  CheckIfAlmostEnd(step_id);
  reach_to_last_alloc = IsLastAlloc();
}

void BFCStatus::DeallocationUpdate(long long bytes_in_use, int step_id, size_t total_region_allocated_bytes_) {
  ++curr_united_id;
  ++deallocation_id;
  chunk_bytes = bytes_in_use;
  region_bytes = total_region_allocated_bytes_;
  reach_to_last_dealloc = IsLastDealloc();
}

// Deprecated
void BFCStatus::RecordRegionAllocEvent(size_t region_size) {
  unsigned long long ts = Env::Default()->NowMicros() - start_time;
  curr_region_events.push_back(RegionEvent(0, is_forward, region_event_idx++, ts, region_size, region_bytes));
  printf(
      "RegionEvent,Alloc,%s,%d,is_forward,%d,idx,%d,event_time,%llu,region_size,%zu,memory_usage,%zu\n",
      name.c_str(), current_step_id, is_forward, region_event_idx, ts / 1000,
      region_size / bytes_to_mb, region_bytes / bytes_to_mb);
  fflush(stdout);
}

// Deprecated
void BFCStatus::RecordRegionFreeEvent(size_t region_size) {
  unsigned long long ts = Env::Default()->NowMicros() - start_time;
  curr_region_events.push_back(RegionEvent(1, is_forward, region_event_idx++, ts, region_size, region_bytes));
  printf(
      "RegionEvent,Free,%s,%d,is_forward,%d,idx,%d,event_time,%llu,region_size,%zu,memory_usage_,%zu\n",
      name.c_str(), current_step_id, is_forward, region_event_idx, ts/1000,
      region_size / bytes_to_mb, region_bytes / bytes_to_mb);
  fflush(stdout);
}

void BFCStatus::IsolatedProfiling(size_t total_region_allocated_bytes_, int alloc_or_dealloc) {
  unsigned long long current_ms = (Env::Default()->NowMicros() - start_time)/1000;
  if (current_ms > isolated_last_ms) {
    if (current_ms == isolated_last_ms + 1) {
      isolated_profiling_info.push_back(std::make_pair(curr_united_id, total_region_allocated_bytes_));
      isolated_profiling_id++;
      // printf("IsolatedProfiling,%s(step%d),%s,current_ms,%llu,isolated_last_ms,%llu,idx,%lld,region_bytes,%zu\n",
      //         name.c_str(), current_step_id, caller.c_str(),
      //         current_ms, isolated_last_ms,
      //         isolated_profiling_id,
      //         total_region_allocated_bytes_/bytes_to_mb);
    }
    else {
      for (int i = 0; i < current_ms - isolated_last_ms - 1; i++) {
        isolated_profiling_info.push_back(std::make_pair(curr_united_id, isolated_last_memory));
        isolated_profiling_id++;
        // printf("IsolatedProfiling,%s(step%d),%s,current_ms,%llu,last_ms,%llu,idx,%lld,region_bytes,%zu\n",
        //         name.c_str(), current_step_id, "fillup",
        //         current_ms, isolated_last_ms,
        //         isolated_profiling_id,
        //         isolated_last_memory/bytes_to_mb);
      }
    }
    isolated_last_ms = current_ms;
    isolated_last_memory = total_region_allocated_bytes_;
  }
}


void BFCStatus::IsolatedProfiling_2(size_t total_region_allocated_bytes_, int alloc_or_dealloc) {
  // 아래 if 문이 필요한 이유는 multiple iteartion에서 profiling을 할 것이기 때문에.
  if (first_isolated_profiling_2) {
    isolated_profiling_start_time_2 = Env::Default()->NowMicros();
    first_isolated_profiling_2 = false;
  }
  // unsigned long long current_ms = (Env::Default()->NowMicros() - start_time)/1000;
  unsigned long long current_ms = (Env::Default()->NowMicros() - isolated_profiling_start_time_2)/1000;
  if (current_ms > isolated_last_ms_2) {
    if (current_ms == isolated_last_ms_2 + 1) {
      isolated_profiling_info_2.push_back(std::make_pair(curr_united_id, total_region_allocated_bytes_));
      isolated_profiling_id_2++;
      // printf("IsolatedProfiling_2,%s(step%d),%s,current_ms,%llu,isolated_last_ms_2,%llu,idx,%lld,region_bytes,%zu\n",
      //         name.c_str(), current_step_id, caller.c_str(),
      //         current_ms, isolated_last_ms_2,
      //         isolated_profiling_id_2,
      //         total_region_allocated_bytes_/bytes_to_mb);
    }
    else {
      for (int i = 0; i < current_ms - isolated_last_ms_2 - 1; i++) {
        isolated_profiling_info_2.push_back(std::make_pair(curr_united_id, isolated_last_memory_2));
        isolated_profiling_id_2++;
        // printf("IsolatedProfiling_2,%s(step%d),%s,current_ms,%llu,last_ms,%llu,idx,%lld,region_bytes,%zu\n",
        //         name.c_str(), current_step_id, (caller+"_fillup").c_str(),
        //         current_ms, isolated_last_ms_2,
        //         isolated_profiling_id_2,
        //         isolated_last_memory_2/bytes_to_mb);
      }
    }
    isolated_last_ms_2 = current_ms;
    isolated_last_memory_2 = total_region_allocated_bytes_;
  }
}

void BFCStatus::ColocatedProfiling(size_t total_region_allocated_bytes_, int alloc_or_dealloc) {
  unsigned long long current_ms = (Env::Default()->NowMicros() - start_time)/1000;
  if (current_ms > colocated_last_ms) {
    if (current_ms == colocated_last_ms + 1) {
      curr_colocated_profiling_info.push_back(std::make_pair(curr_united_id, total_region_allocated_bytes_));
      colocated_profiling_id++;
      // printf("ColocatedProfiling,%s,%d,%s,current_ms,%llu,last_ms,%llu,idx,%lld,region_bytes,%zu\n",
      //         name.c_str(), current_step_id, caller.c_str(),
      //         current_ms, colocated_last_ms,
      //         colocated_profiling_id,
      //         total_region_allocated_bytes_/bytes_to_mb);
    } else {
      for (int i = 0; i < current_ms - colocated_last_ms - 1; i++) {
        curr_colocated_profiling_info.push_back(std::make_pair(curr_united_id, colocated_last_memory));
        colocated_profiling_id++;
        // printf("ColocatedProfiling,%s,%d,%s,current_ms,%llu,colocated_last_ms,%llu,idx,%lld,region_bytes,%zu\n",
        //         name.c_str(), current_step_id, "fillup",
        //         current_ms, colocated_last_ms,
        //         colocated_profiling_id,
        //         colocated_last_memory/bytes_to_mb);
      }
    }
    colocated_last_ms = current_ms;
    colocated_last_memory = total_region_allocated_bytes_;
  }
}

cr_GlobalPool::cr_GlobalPool() {
  free_regions.reserve(100000);
}

cr_GlobalPool::~cr_GlobalPool() {}

void cr_GlobalPool::RecordCUDA(std::string name, int step_id, size_t bytes, bool sharable_, bool is_backward, std::string mem_op) {
  mutex_lock l(lock_);
  total_aggregated_alloc_memory += bytes;
  // UpdateRemainingBudget("RecordCUDAAlloc");
  // UpdateAvailableGPUMemory("RecordCUDAAlloc");
  printf("RecordCUDA%s,%s,%d,bytes,%zu,KB,sharable,%d,is_backward,%d,total_aggregated_alloc_memory,%lld,available_gpu_memory,%lld\n",
         mem_op.c_str(),
         name.c_str(), 
         step_id, 
         bytes / 1024,
         sharable_,
         is_backward,
         total_aggregated_alloc_memory / bytes_to_mb,
         available_gpu_memory / bytes_to_mb);
  fflush(stdout);
}

// void cr_GlobalPool::RecordCUDAAlloc(std::string name, int step_id, size_t bytes, bool sharable_, bool is_backward) {
//   mutex_lock l(lock_);
//   total_aggregated_alloc_memory += bytes;
//   UpdateRemainingBudget("RecordCUDAAlloc");
//   UpdateAvailableGPUMemory("RecordCUDAAlloc");
//   printf("RecordCUDAAlloc,%s,%d,bytes,%zu,KB,sharable,%d,total_aggregated_alloc_memory,%zu,available_gpu_memory,%zu\n",
//          name.c_str(), 
//          step_id, 
//          bytes / 1024,
//          sharable_,
//          total_aggregated_alloc_memory / bytes_to_mb,
//          available_gpu_memory / bytes_to_mb);
// }

// void cr_GlobalPool::RecordCUDAFree(std::string name, int step_id, size_t bytes, bool sharable_, bool is_backward) {
//   mutex_lock l(lock_);
//   total_aggregated_alloc_memory -= bytes;
//   UpdateRemainingBudget("RecordCUDAFree");
//   UpdateAvailableGPUMemory("RecordCUDAFree");
//   printf("RecordCUDAFree,%s,%d,bytes,%zu,KB,sharable,%d,total_aggregated_alloc_memory,%zu, "
//          "available_gpu_memory,%zu\n",
//          name.c_str(), 
//          step_id, 
//          bytes / 1024,
//          sharable_,
//          total_aggregated_alloc_memory / bytes_to_mb,
//          available_gpu_memory / bytes_to_mb);
// }

void cr_GlobalPool::UpdateRemainingBudget(std::string caller) {
  if (memory_budget - total_aggregated_alloc_memory >= 0) {
    buffer_touch = false;
    remaining_memory_budget = memory_budget - total_aggregated_alloc_memory;
  } else {
    buffer_touch = true;
    remaining_memory_budget = 0;
    printf("BufferTouch!\n", caller.c_str());
  }
}

void cr_GlobalPool::UpdateAvailableGPUMemory(std::string caller) {
  if (gpu_memory_limit - total_aggregated_alloc_memory >= 0) {
    host_memory_touch = false;
    available_gpu_memory = gpu_memory_limit - total_aggregated_alloc_memory; // == remaining_memory_budget + buffer_size
    touched_host_memory = 0;
  } else {
    host_memory_touch = true;
    available_gpu_memory = 0;
    touched_host_memory = total_aggregated_alloc_memory - gpu_memory_limit;
    printf("HostMemoryTouch!,%lldMB\n", caller.c_str(), touched_host_memory/bytes_to_mb);
  }
}

void cr_GlobalPool::AddNewBFCStatus(int bfc_id, std::string bfc_name) {
  CHECK_NE(bfc_id, -1);
  CHECK_NE(bfc_name, "cuda_host_bfc");
  BFCStatus bfc_status(bfc_id, bfc_name, start_aligning_step, num_isolated_profiling, this);
  printf("cr_GlobalPool,AddNewBFCStatus(%d, %s)\n", bfc_id, bfc_name.c_str());
  fflush(stdout);
  bfc_status_vec.insert(std::pair<int, BFCStatus>(bfc_id, bfc_status));
  ++num_allocators;
}

void cr_GlobalPool::PrintRegionMove(int step_id, std::string action, std::string src, std::string dst, Region region, 
                                    size_t total_region_allocated_bytes, unsigned long long elapsed, bool sharable_, 
                                    bool is_forward) {
  printf("%s,%d,%s,%s,%s,%zu,%d,%d,-%zu,%zu,%lld,%llu\n", 
          "RegionMove",
          step_id, 
          action.c_str(), 
          src.c_str(), 
          dst.c_str(), 
          region.region_id, 
          sharable_,
          is_forward, 
          region.size/bytes_to_mb,
          total_region_allocated_bytes/bytes_to_mb, 
          aggregated_region_bytes/bytes_to_mb,
          elapsed);
  // fflush(stdout);
  // std::string str = "RegionMove,"
  //                   + std::to_string(step_id) + ","
  //                   + action + ","
  //                   + src + ","
  //                   + dst + ","
  //                   + std::to_string(region.region_id) + ","
  //                   + std::to_string(sharable_) + ","
  //                   + std::to_string(is_forward) + ","
  //                   + std::to_string(region.size/bytes_to_mb) + ","
  //                   + std::to_string(total_region_allocated_bytes/bytes_to_mb) + ","
  //                   + std::to_string(aggregated_region_bytes/bytes_to_mb) + ","
  //                   + std::to_string(elapsed);
  // std::ostringstream os;
  // os << "RegionMove" + ","
  //       + std::to_string(step_id) + ","
  //       + "Alloc" + ","
  //       + src + ","
  //       + dst + ","
  //       + std::to_string(region.region_id) + ","
  //       + std::to_string(sharable_) + ","
  //       + std::to_string(is_forward) + ","
  //       + std::to_string(region.size/bytes_to_mb) + ","
  //       + std::to_string(total_region_allocated_bytes/bytes_to_mb) + ","
  //       + std::to_string(aggregated_region_bytes/bytes_to_mb) + ","
  //       + std::to_string(elapsed);
  // std::string str(os.str());
  // parser.file_write(str);
}

void cr_GlobalPool::InitGlobalPool(unsigned long long base_micro, size_t memory_limit_) {
  if (!initialized) {
    print_bfc = parser.parse_target_config("PrintBFC");
    print_bfc_step = parser.parse_target_config("PrintBFCStep");
    tight_fit = parser.parse_target_config("TightFit");
    print_period = parser.parse_target_config("PrintPeriod");
    coordinator_policy = parser.parse_string_config("CoordinatorPolicy");
    turn_on_coordinator = parser.parse_target_config("TurnOnCoordinator");
    num_virtual_gpu = parser.parse_target_config("NumVirtualGPU");
    start_aligning_step = parser.parse_target_config("StartAligningStep");
    if (ModelBasedCoordination()) {
      use_colocated_simulator = parser.parse_target_config("UseColocatedSimulator");
    } else {
      use_colocated_simulator = false;
    }
    base_micro = base_micro;
    gpu_memory_limit = memory_limit_;
    buffer_ratio = parser.parse_target_config("BufferRatio");
    ResetMemoryComposition(buffer_ratio, 0);
    printf("gpu_memory_limit,%lld,memory_budget,%lld,buffer_size,%lld,buffer_ratio,%lld\n",
           gpu_memory_limit / bytes_to_mb, memory_budget / bytes_to_mb,
           buffer_size / bytes_to_mb, buffer_ratio);
    std::cout<<"RegionMove,Action,StepId,Src,Dst,RegionId,Sharable,IsForward,RegionSize,IndividualRegionBytes,AggRegionBytes,Time"<<std::endl;
    printf("RegionEvent,Alloc,Allocator,StepId,IsForward,RegionEventIdx,EventTime,RegionSize,RegionUsage\n");
    fflush(stdout);
    initialized = true;
  }
  else {
    printf("cr_GlobalPool::InitGlobalPool. cr_GlobalPool has been initialized already.\n");
  }
}

/*
	Local BFC ====<AllocationRegion>====> Global Pool
*/
// bool cr_GlobalPool::FreeRegionToGlobalPool(void* memory_ptr, size_t freed_region_size,
//                                   size_t region_id, bool sharable_,
//                                   const std::string src_allocator_name,
//                                   int step_id,
//                                   unsigned long long elapsed, int bfc_id, 
//                                   size_t total_region_allocated_bytes_) {
//   mutex_lock l(lock_);
//   global_pool_holding_bytes += freed_region_size;
//   // Creating new FreeAR object with given information
//   Region new_free_ar;
//   new_free_ar.region_id = region_id;
//   new_free_ar.size = freed_region_size;
//   new_free_ar.sharable = sharable_;
//   new_free_ar.allocator_name = src_allocator_name;
//   new_free_ar.memory_ptr = memory_ptr;
// 	// Sorting해서 가지고 있어야하나. 가장 작은 거 중에 fit하는 걸 먼저 return 하기 위해
// 	// first fit
//   int i = 0;
//   bool inserted = false;
//   for (std::vector<Region>::iterator it = free_regions.begin(); it != free_regions.end(); ++it) {
//     if (it->size >= freed_region_size) {
//       free_regions.insert(it, new_free_ar);
//       inserted = true;
//       break;
//     }
//   }
//   if (!inserted) {
//     free_regions.push_back(new_free_ar);
//   }

//   // bfc_status_vec[bfc_id].region_bytes = total_region_allocated_bytes_ - freed_region_size;
//   // PrintRegionMove(step_id, src_allocator_name, "Free", "cr_GlobalPool", new_free_ar,
//   //                 bfc_status_vec[bfc_id].region_bytes, elapsed, sharable_,
//   //                 bfc_status_vec[bfc_id].is_forward);
//   // bfc_status_vec[bfc_id].RecordRegionFreeEvent(freed_region_size);

//   return true;
// }

bool cr_GlobalPool::FreeRegionToGlobalPool(void* memory_ptr, size_t freed_region_size,
                                  size_t region_id, bool sharable_,
                                  const std::string src_allocator_name,
                                  int step_id,
                                  unsigned long long elapsed, int bfc_id, 
                                  size_t total_region_allocated_bytes_, 
                                  ChunkHandle* handles_,
                                  unsigned long long birth_time) {
  bool only_one_size_region = false;
  if (only_one_size_region) {
    mutex_lock l(lock_);
    global_pool_holding_bytes += freed_region_size;
    CHECK_GE(global_pool_holding_bytes, 0);
    Region new_free_ar;
    new_free_ar.region_id = region_id;
    new_free_ar.size = freed_region_size;
    new_free_ar.sharable = sharable_;
    new_free_ar.allocator_name = src_allocator_name;
    new_free_ar.memory_ptr = memory_ptr;
    new_free_ar.alloc_time = 0;
    new_free_ar.handles = handles_;
    free_regions.push_back(new_free_ar);
    if (print_bfc == 1) {
      PrintRegionMove(step_id, src_allocator_name, "Free", "cr_GlobalPool", new_free_ar,
                      bfc_status_vec[bfc_id].region_bytes, elapsed, sharable_,
                      bfc_status_vec[bfc_id].is_forward);
    }
    return true;
  }
  else {
    mutex_lock l(lock_);
    global_pool_holding_bytes += freed_region_size;
    CHECK_GE(global_pool_holding_bytes, 0);
    // Creating new FreeAR object with given information
    Region new_free_ar;
    new_free_ar.region_id = region_id;
    new_free_ar.size = freed_region_size;
    new_free_ar.sharable = sharable_;
    new_free_ar.allocator_name = src_allocator_name;
    new_free_ar.memory_ptr = memory_ptr;
    new_free_ar.alloc_time = 0;
    new_free_ar.handles = handles_;
    // Sorting해서 가지고 있어야하나. 가장 작은 거 중에 fit하는 걸 먼저 return 하기 위해
    // first fit
    int i = 0;
    bool inserted = false;
    for (std::vector<Region>::iterator it = free_regions.begin(); it != free_regions.end(); ++it) {
      if (it->size > freed_region_size) {
        free_regions.insert(it, new_free_ar);
        inserted = true;
        break;
      }
    }
    if (!inserted) {
      free_regions.push_back(new_free_ar);
    }

    // bfc_status_vec[bfc_id].region_bytes = total_region_allocated_bytes_ - freed_region_size;
    if (print_bfc == 1) {
      PrintRegionMove(step_id, src_allocator_name, "Free", "cr_GlobalPool", new_free_ar,
                      bfc_status_vec[bfc_id].region_bytes, elapsed, sharable_,
                      bfc_status_vec[bfc_id].is_forward);
    }

    // bfc_status_vec[bfc_id].RecordRegionFreeEvent(freed_region_size);

    unsigned long long death_time = Env::Default()->NowNanos();
    unsigned long long lifetime = Env::Default()->NowNanos() - birth_time;
    std::tuple<void*, int, uint64, uint64, uint64> tup = std::make_tuple(memory_ptr, region_id, birth_time, death_time, lifetime);
    bfc_status_vec[bfc_id].region_lifetime.push_back(tup);
    return true;
  }
}

Region cr_GlobalPool::NullRegion() {
  Region null_region;
  null_region.region_id = 0;
  null_region.size = 0;
  null_region.sharable = false;
  null_region.allocator_name = "null_region";
  null_region.memory_ptr = nullptr;
  null_region.handles = nullptr;
  return null_region;
}

/*
  Global Pool ====<AllocationRegion>====> Local BFC
*/
Region cr_GlobalPool::AllocRegionFromGlobalPool(size_t rounded_bytes, size_t minimum_required_region_size,
                                                const std::string dst_allocator_name, int step_id,
                                                unsigned long long elapsed, bool sharable_,
                                                size_t total_region_allocated_bytes_, int bfc_id) {
  bool only_one_size_region = false;
  if (only_one_size_region) {
    mutex_lock l(lock_);
    if (free_regions.empty()) {
      printf("FAIL,%s,%d,%zu, GlobalPool is empty. (Causing CUDA Alloc)\n", dst_allocator_name.c_str(), step_id, total_region_allocated_bytes_);
      return NullRegion();
    }
    std::vector<Region>::iterator it = free_regions.begin();
    Region ret_free_ar = *it;
    global_pool_holding_bytes -= it->size;
    if (print_bfc == 1) {
      PrintRegionMove(step_id, "Alloc", "cr_GlobalPool", dst_allocator_name, *it,
                      bfc_status_vec[bfc_id].region_bytes, elapsed,
                      sharable_, bfc_status_vec[bfc_id].is_forward);
    }
    free_regions.erase(it);
    return ret_free_ar;
  }
  else {
    mutex_lock l(lock_);
    if (free_regions.empty()) {
      printf("FAIL,%s,%d,%zu, GlobalPool is empty. (Causing CUDA Alloc)\n", dst_allocator_name.c_str(), step_id, total_region_allocated_bytes_);
      return NullRegion();
    }
    else {
      while (true) {
        for (std::vector<Region>::iterator it = free_regions.begin(); it != free_regions.end(); ++it) {
          if (it->size >= rounded_bytes) {
            if (tight_fit == 0 || (tight_fit == 1 && it->size <= minimum_required_region_size * 2)) {
              Region ret_free_ar = *it;
              ret_free_ar.alloc_time = Env::Default()->NowMicros();
              global_pool_holding_bytes -= it->size;
              CHECK_GE(global_pool_holding_bytes, 0);
              if (print_bfc == 1) {
                PrintRegionMove(step_id, "Alloc", "cr_GlobalPool", dst_allocator_name, *it,
                                bfc_status_vec[bfc_id].region_bytes, elapsed,
                                sharable_, bfc_status_vec[bfc_id].is_forward);
              }
              free_regions.erase(it);
              if (bfc_status_vec[bfc_id].cuda_alloc_is_delayed) {
                bfc_status_vec[bfc_id].cuda_alloc_delay_end = Env::Default()->NowMicros() - bfc_status_vec[bfc_id].start_time;
                bfc_status_vec[bfc_id].cuda_alloc_delay_time = bfc_status_vec[bfc_id].cuda_alloc_delay_end - bfc_status_vec[bfc_id].cuda_alloc_delay_start;
                printf("%s,%d,AllocRegionFromGlobalPool,cuda_alloc_delay_end,cuda_alloc_delay_time,%llu\n",
                      dst_allocator_name.c_str(), step_id, bfc_status_vec[bfc_id].cuda_alloc_delay_time / 1000);
              }
              // bfc_status_vec[bfc_id].RecordRegionAllocEvent(it->size);

              return ret_free_ar; // while loop break point 1
            }
            else {
              printf("FAIL,GlobalPoolAlloc,%s,%d,,BestFit: %zu, RequiredSize: %zu, (It will cause an additional CUDA Alloc.)\n", 
                      dst_allocator_name.c_str(), step_id, it->size/bytes_to_mb, minimum_required_region_size/bytes_to_mb);
              break; // for loop break
            }
          }
        }
        printf("FAIL,GetRegionFromGlobalPool,%s,%d,RequestedRegionSize,%zu MB.\n", dst_allocator_name.c_str(), step_id, minimum_required_region_size/(1024*1024));
        printf("FAIL,GlobalPool has (");
        for (std::vector<Region>::iterator it = free_regions.begin(); it != free_regions.end(); ++it) {
          printf("%zu MB, ", it->size/(1024*1024));
        }
        printf(")\n");
        /*
          Sharable Region만 share하는게 아니라 전부 쉐어를 하면 백워드에서 Region
          Alloc이 일어날 수도 있고 그러면 Deadlock이 걸리기 때문에 필요한 사이즈의
          Region이 없을 시 while 루프를 돌지 않고 바로 NullRegion을 리턴한다.
        */
        if (num_virtual_gpu < 2) {
          return NullRegion();
        }
        if(TryCudaAlloc(minimum_required_region_size, dst_allocator_name, bfc_id, step_id)) {
          return NullRegion();
        }
        else {
          if (bfc_status_vec[bfc_id].cuda_alloc_is_delayed == false) {
            printf("%s,%d,CUDAAllocDelay\n", dst_allocator_name.c_str(), step_id);
            bfc_status_vec[bfc_id].cuda_alloc_is_delayed = true;
            bfc_status_vec[bfc_id].cuda_alloc_delay_start = Env::Default()->NowMicros();
          }
          Env::Default()->SleepForMicroseconds(1000);
        }
      }
    }
  }
}



bool cr_GlobalPool::IfHostMemoryTouched() { return host_memory_touch; }

void cr_GlobalPool::BFC1_Sleep_Until_BFC0_Finishes_IsolatedProfiling(int step_id) {
  /*
    GPU_1_bfc, step 15
    step 15 시작포인트에서 GPU_0_bfc 의 Isolated Profiling(step24)가 끝나길 기다린다.
  */
  bool print_once = true;
  while (true) {
    if (bfc_status_vec[0].finish_isolated_profiling) {
      printf("DAMN,%s(step%d) Finishes waiting for %s's end of isolated profiling\n",
            bfc_status_vec[1].name.c_str(), step_id,
            bfc_status_vec[0].name.c_str());
      break;
    }
    else {
      if (print_once) {
        printf("DAMN,%s(step%d) Waits for %s's end of isolated profiling\n",
              bfc_status_vec[1].name.c_str(), step_id,
              bfc_status_vec[0].name.c_str());
        print_once = false;
      }
    }
  }
}

void cr_GlobalPool::BFC0_Sleep_Until_BFC1_Finishes_IsolatedProfiling(int step_id) {
  bool print_once = true;
  while (true) {
    if (bfc_status_vec[1].finish_isolated_profiling) {
      printf("DAMN,%s(step%d) Finishes waiting for %s's isolated profiling\n",
            bfc_status_vec[0].name.c_str(), step_id,
            bfc_status_vec[1].name.c_str());
      break;
    }
    else {
      if (print_once) {
        printf("DAMN,%s(step%d) Waits for %s's isolated profiling\n",
              bfc_status_vec[0].name.c_str(), step_id,
              bfc_status_vec[1].name.c_str());
        print_once = false;
      }
    }
  }
}

void cr_GlobalPool::FirstAligning(int step_id) {
  bool print_once = true;
  while (true) {
    if (bfc_status_vec[0].is_backward == true &&
        bfc_status_vec[0].is_forward == false &&
        bfc_status_vec[0].almost_end == true &&
        bfc_status_vec[0].current_step_id == start_aligning_step + num_isolated_profiling) {
      printf("DAMN,%s(step%d) Aligns with %s almost_end(AlmostEnd).\n",
              bfc_status_vec[1].name.c_str(),
              bfc_status_vec[1].current_step_id,
              bfc_status_vec[0].name.c_str());
      break;
    }
    else {
      if (print_once) {
        print_once = false;
        printf("DAMN,%s(step%d) Waits for %s almost_end(AlmostEnd).\n",
              bfc_status_vec[1].name.c_str(),
              bfc_status_vec[1].current_step_id,
              bfc_status_vec[0].name.c_str());
      }
    }
  }
}

void cr_GlobalPool::SimulateIfNewStep(int bfc_id, int step_id) {

  // mutex_lock l(lock_);

  // if (bfc_status_vec[bfc_id].IsNewGlobalStep(step_id)) {
    // bfc_status_vec[bfc_id].EndOfOldStep(step_id); // start time 이자 end time 기록하는 부분.
    // if (ModelBasedCoordination()) {

      while(true) {
        if (bfc_status_vec[0].current_step_id >= (start_aligning_step - 1) && 
            bfc_status_vec[1].current_step_id >= (start_aligning_step - 1)) {
          break;
        }
      }

      /* GPU_1_bfc는 step 15 시작포인트에서 GPU_0_bfc 의 Isolated Profiling(step24)가 끝나길 기다린다. */
      if (bfc_id == 1 && step_id == start_aligning_step) {
        BFC1_Sleep_Until_BFC0_Finishes_IsolatedProfiling(step_id);
      }
      /* GPU_0_bfc는 step 25 시작포인트에서 GPU_1_bfc 의 Isolated Profiling(step24)가 끝나길 기다린다. */
      if (bfc_id == 0 && step_id == start_aligning_step + num_isolated_profiling ) {
        BFC0_Sleep_Until_BFC1_Finishes_IsolatedProfiling(step_id);
      }
      if (bfc_id == 1 && step_id == start_aligning_step + num_isolated_profiling) {
        FirstAligning(step_id);
      }
      // if (step_id >= start_aligning_step + num_isolated_profiling) {
      if (step_id > start_aligning_step + num_isolated_profiling) {
        // if (IfHostMemoryTouched()) {
        //   IncreaseBufferSize(step_id);
        // }
        Simulate(bfc_id, step_id);
      }
    // }
    // bfc_status_vec[bfc_id].ActualStartOfNewStep();
  // }
}

bool cr_GlobalPool::Simulate(int bfc_id, int step_id) {
  bool print_once = true;
  while (true) {
    if (print_once) {
      print_once = false;
      // printf("DAMN,Simulate,Start,%s,%d\n", bfc_status_vec[bfc_id].name.c_str(), bfc_status_vec[bfc_id].current_step_id);
    }
    bool ret_sim = false;
    // if (step_id >= start_aligning_step + num_isolated_profiling) {
      if (bfc_id == 0 && bfc_status_vec[1].curr_united_id == 0) {
        break;
      } 
      else {
        if (coordinator_policy == "ModelBased") {
          if (CondForIsolatedSimulator(step_id)) {
            // unsigned long long ts = Env::Default()->NowNanos();
            ret_sim = IsolatedSimulator(bfc_id, step_id);
            // printf("%s,%d,SchedulingOverhead,%llu\n", bfc_status_vec[bfc_id].name.c_str(), step_id, (Env::Default()->NowNanos() - ts));
          }
          else if (CondForIsolatedSimulator_2(step_id)) {
            // ret_sim = IsolatedSimulator_2(bfc_id, step_id);
            ret_sim = IsolatedSimulator(bfc_id, step_id);
          }
          else {
            printf("GANGMUK_ERROR: Unexpected behavior\n");
            fflush(stdout);
            CHECK_EQ(0,1);
          }
          // ret_sim = ColocatedSimulator(bfc_id, step_id);
        }
        else if (coordinator_policy == "Spatial") {
          ret_sim = SpatialSimulator(bfc_id, step_id);
        }
        else if (coordinator_policy == "TickTock") {
          ret_sim = TickTockScheduler(bfc_id, step_id);
        }
        else {
          printf("GANGMUK_ERROR: Unexpected coordinator_policy: %s (global_pool.cc)\n", coordinator_policy.c_str());
          fflush(stdout);
          CHECK_EQ(0,1);
        }
      }
    // }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    if (ret_sim) {
      bool fair_scheduling_option = false;
      if (fair_scheduling_option) {
        while(true) {
          int other_id = 0;
          if (bfc_id == 0) {
            other_id = 1;
          }
          if (bfc_id == 0) {
            if ((bfc_status_vec[bfc_id].current_step_id - 1) == bfc_status_vec[other_id].current_step_id) {
              break;
            }
          }
          else if (bfc_id == 1) {
            if (bfc_status_vec[bfc_id].current_step_id == bfc_status_vec[other_id].current_step_id) {
              break;
            }
          }
        }
      }
      bfc_status_vec[bfc_id].total_timeshift = bfc_status_vec[bfc_id].num_timeshift * timeshift_unit;
      // printf("DAMN,%s,Done,%s,%d,num_timeshift,%llu,total_delay_time,%llu,ms\n",
      //         coordinator_policy.c_str(),
      //         bfc_status_vec[bfc_id].name.c_str(),
      //         bfc_status_vec[bfc_id].current_step_id,
      //         bfc_status_vec[bfc_id].num_timeshift,
      //         bfc_status_vec[bfc_id].total_timeshift / 1000);
      break;
    }
    else {
      Env::Default()->SleepForMicroseconds(timeshift_unit);  // 0.5ms sleep
      ++bfc_status_vec[bfc_id].num_timeshift;
    }
  } // while
}

bool cr_GlobalPool::IsolatedSimulator(int bfc_id, int step_id) {
  int other_id = 0;
  if (bfc_id == 0) {
    other_id = 1;
  }
  BFCStatus& me = bfc_status_vec[bfc_id];
  BFCStatus& other = bfc_status_vec[other_id];
  int other_curr_idx = other.GetCurrIsolatedProfilingIdx();
  int last_idx = std::min(me.isolated_profiling_info.size(), other.isolated_profiling_info.size() - other_curr_idx);
  for (int me_curr_idx = 0; me_curr_idx < last_idx; ++me_curr_idx, ++other_curr_idx) {
    size_t me_estimation = me.isolated_profiling_info[me_curr_idx].second;
    size_t other_estimation = other.isolated_profiling_info[other_curr_idx].second;
    if (me_estimation + other_estimation > memory_budget) {
      // FAIL
      // PrintSimulateResult("IsolatedSimulator", "Fail", step_id, me, other, me_estimation, other_estimation);
      return false;
    }
    else {
      // PASS one test
      // PrintSimulateResult("IsolatedSimulator", "Pass", step_id, me, other, me_estimation, other_estimation);
    }
  }
  // Now, PASS All test
  // printf("IsolatedSimulator,Success,%s,step,%d\n", me.name.c_str(), step_id);
  return true;
}

bool cr_GlobalPool::IsolatedSimulator_2(int bfc_id, int step_id) {
  int other_id = 0;
  if (bfc_id == 0) {
    other_id = 1;
  }
  BFCStatus& me = bfc_status_vec[bfc_id];
  BFCStatus& other = bfc_status_vec[other_id];
  int other_curr_idx = other.GetCurrIsolatedProfilingIdx_2();
  int last_idx = std::min(me.isolated_profiling_info_2.size(), other.isolated_profiling_info_2.size() - other_curr_idx);
  for (int me_curr_idx = 0; me_curr_idx < last_idx; ++me_curr_idx, ++other_curr_idx) {
    size_t me_estimation = me.isolated_profiling_info_2[me_curr_idx].second;
    size_t other_estimation = other.isolated_profiling_info_2[other_curr_idx].second;
    if (me_estimation + other_estimation > memory_budget) {
      // FAIL
      // PrintSimulateResult("IsolatedSimulator_2", "Fail", step_id, me, other, me_estimation, other_estimation);
      return false;
    }
    else {
      // PASS one test
      // PrintSimulateResult("IsolatedSimulator_2", "Pass", step_id, me, other, me_estimation, other_estimation);
    }
  }
  // Now, PASS All test
  // printf("IsolatedSimulator_2,Success,%s,step,%d\n", me.name.c_str(), step_id);
  return true;
}

bool cr_GlobalPool::SpatialSimulator(int bfc_id, int step_id) {
  int other_id = 0;
  if (bfc_id == 0) {
    other_id = 1;
  }
  BFCStatus& me = bfc_status_vec[bfc_id];
  BFCStatus& other = bfc_status_vec[other_id];
  size_t me_peak_memory = me.region_peak_bytes;
  size_t other_peak_memory = 0;
  if (other.is_backward) {
    other_peak_memory = std::max(other.region_bytes, other.backward_region_peak_bytes);
  } else {
    other_peak_memory = other.region_peak_bytes;
  }
  if (me_peak_memory + other_peak_memory > memory_budget) {
    // printf("SpatialSimulator,Fail,%s,step,%d,%s,%zu,%s,%zu,budget,%zu\n", 
    //         me.name.c_str(), step_id, 
    //         me.name.c_str(), me_peak_memory / bytes_to_mb, 
    //         other.name.c_str(), other_peak_memory / bytes_to_mb,  
    //         memory_budget / bytes_to_mb);
    return false;
  }
  else {
    printf("SpatialSimulator,Success,%s,step,%d\n", me.name.c_str(), step_id);
    return true;
  }
}

bool cr_GlobalPool::TickTockScheduler(int bfc_id, int step_id) {
  int other_id = 0;
  if (bfc_id == 0) {
    other_id = 1;
  }
  BFCStatus& me = bfc_status_vec[bfc_id];
  BFCStatus& other = bfc_status_vec[other_id];
  if (other.is_backward) {
    printf("TickTockScheduler,Success,%s,step,%d\n", me.name.c_str(), step_id);
    return true;
  }
  else {
    printf("TickTockScheduler,Fail(Waiting),%s,step,%d\n", me.name.c_str(), step_id);
    return false;
  }

  ///////////////////////////////////////////////////////////
  // int other_id = 0;
  // if (bfc_id == 0) {
  //   other_id = 1;
  // }
  // BFCStatus& me = bfc_status_vec[bfc_id];
  // BFCStatus& other = bfc_status_vec[other_id];
  // if (bfc_id == 0) {
  //   if (me.curr_united_id == me.region_peak_allocation_id) {
  //     while (true) {
  //       if (other.is_waiting) {
  //         me.synced = true;
  //         break;
  //       }
  //     }
  //   }
  // }
  ///////////////////////////////////////////////////////////
}

bool cr_GlobalPool::ColocatedSimulator(int bfc_id, int step_id) {
  int other_id = 0;
  if (bfc_id == 0) {
    other_id = 1;
  }
  BFCStatus& me = bfc_status_vec[bfc_id];
  BFCStatus& other = bfc_status_vec[other_id];
  int other_start_idx = 0;
  for (int i = 0; i < other.prev_prev_colocated_profiling_info.size(); ++i) {
    if (other.prev_prev_colocated_profiling_info[i].first >= bfc_status_vec[other_id].curr_united_id) {
      other_start_idx = i;
      break;
    }
  }
  int last_idx = -1;
  if (other.prev_prev_colocated_profiling_info.size() - other_start_idx > me.prev_prev_colocated_profiling_info.size()) {
    last_idx = me.prev_prev_colocated_profiling_info.size();
  } else {
    last_idx = other.prev_prev_colocated_profiling_info.size() - other_start_idx;
  }
  int j = other_start_idx;
  for (int i = 0; i < last_idx; ++i, ++j) {
    size_t me_estimation = me.prev_prev_colocated_profiling_info[i].second;
    size_t other_estimation = other.prev_prev_colocated_profiling_info[j].second;
    if (me_estimation + other_estimation > memory_budget) {
      PrintSimulateResult("ColocatedSimulator", "Fail", step_id, me, other, me_estimation, other_estimation);
      return false;
    }
    PrintSimulateResult("ColocatedSimulator", "Pass", step_id, me, other, me_estimation, other_estimation);
  }
  // printf("ColocatedSimulator,Success,%s,step,%d\n", me.name.c_str(), step_id);
  return true;
}

void cr_GlobalPool::IncreaseBufferSize(int step_id) {
  if (buffer_ratio + buffer_control_unit < 100) {
    buffer_ratio += buffer_control_unit;
    ResetMemoryComposition(buffer_ratio, step_id);
  } else {
    printf("ERROR(gangmuk): Buffer size can't excess memory budget.");
    CHECK_EQ(0,1);
  }
}


// void cr_GlobalPool::PrintBFCLog_2(int step_id, int bfc_id, std::string alloc_dealloc, Chunk_MetaData chunk_metadata, long long bytes_in_use, size_t total_region_allocated_bytes_) {
//   ++bfc_status_vec[bfc_id].incrementer;
//   if (bfc_status_vec[bfc_id].incrementer % print_period == 0) {
//     bfc_status_vec[bfc_id].incrementer = 0;
//     if ((print_bfc == 1 && step_id >= print_bfc_step && step_id < (print_bfc_step + 10)) || 
//         (print_bfc == 2 && step_id >= print_bfc_step) || 
//         print_bfc == -1) {
//       if (chunk_metadata.kernel_name != "dummy") {
//         unsigned long long elapsed = Env::Default()->NowMicros() - base_micro;
//         printf("%s,%d,%s,%s,%d,%lld,%zu,%lld,%lld,%zu,%lld,%llu,%d,%d,%d,%d,%llu,%s,%s,%d\n",
//                 "BFCRecord_4", 
//                 step_id, 
//                 alloc_dealloc.c_str(), 
//                 bfc_status_vec[bfc_id].name.c_str(),
//                 chunk_metadata.executorimpl_id,
//                 chunk_metadata.gangmuk_allocation_id, //bfc_status_vec[bfc_id].allocation_id, 이거 버그. 실질적인 allocation기준의 id가 아니다.
//                 chunk_metadata.chunk_size,
//                 bytes_in_use,
//                 aggregated_chunk_bytes,
//                 total_region_allocated_bytes_,
//                 aggregated_region_bytes,
//                 elapsed,
//                 chunk_metadata.is_gradient,
//                 chunk_metadata.sharable,
//                 chunk_metadata.set_output,
//                 chunk_metadata.is_backward, //bfc_status_vec[bfc_id].is_backward, 이거도 버그. 아주 빡치는 버그. 현재가 forward인지 backward인지 가 중요한게 아니라! 
//                 chunk_metadata.lifespan/1000, /* -1 for allocation */
//                 chunk_metadata.caller.c_str(),
//                 chunk_metadata.kernel_name.c_str(),
//                 chunk_metadata.having_bridge);
//           fflush(stdout);
//       }
//     }
//   }
// }

// void cr_GlobalPool::RecordMemOperation(int step_id, int bfc_id, std::string alloc_dealloc, Chunk_MetaData chunk_metadata, long long bytes_in_use, size_t total_region_allocated_bytes_) {
void cr_GlobalPool::RecordMemOperation(int step_id, int bfc_id, int alloc_dealloc, Chunk_MetaData chunk_metadata, long long bytes_in_use, size_t total_region_allocated_bytes_) {
  if (alloc_dealloc == 1) {
    bfc_status_vec[bfc_id].AllocationUpdate(bytes_in_use, step_id, total_region_allocated_bytes_);
    // set_output 태깅은 allocation할 때가 아니라 그 뒤 deallocation이 되기 전 언젠가 이다.
    // is_backward 표시는 allocation 시점에서 하고, deallocation log를 print할 때는 allocation 시점에 chunk에 표시해놨던 값을 읽어서 이게 allocation 되는 시점에 pre-peak memory 였는지 post-peak memory 였는지 보는 것이다.
    // 여기서 또 중요한 건 말한 거처럼, pre-peak인지 post-peak인지, 즉 forward 인지 backward인지는 allocation 시점으로 판단을 해야한다. 왜냐하면 그게 해당 node가 execution 되는 타이밍이니까.
    // chunk_metadata.is_backward = bfc_status_vec[bfc_id].is_backward; // 이걸 안해줬었어서 is_backward를 출력하면 이상한 값이 나왔다. update: 그런데 이거도 버그야.. 결국 진짜 chunk 자체에는 해당 정보가 기록이 안돼. 이건 그냥 local 변수에만 is_backward를 저장하는 거 뿐이야. 그렇기 때문에 deallocation시에는 is_backward였는지에 대한 정보가 없어.
  }
  else if (alloc_dealloc == -1) {
    bfc_status_vec[bfc_id].DeallocationUpdate(bytes_in_use, step_id, total_region_allocated_bytes_);
  }
  else {
    printf("[GANGMUK_ERROR]: Unexpected alloc_dealloc variable: %d\n", alloc_dealloc);
    fflush(stdout);
    CHECK_EQ(0,1);
  }
  UpdateAggChunkBytes();
  UpdateAggRegionBytes();
  // PrintBFCLog_2(step_id, bfc_id, alloc_dealloc, chunk_metadata, bytes_in_use,  total_region_allocated_bytes_);

  if (print_bfc == 1) {
    if (bfc_status_vec[bfc_id].incrementer++ % print_period == 0) {
      bfc_status_vec[bfc_id].incrementer = 0;
      bfc_status_vec[bfc_id].PushBackBFCLog(step_id, alloc_dealloc, chunk_metadata, bytes_in_use,  total_region_allocated_bytes_, global_pool_holding_bytes);
    }
  }
  int other_bfc_id = 0;
  if (bfc_id == 0) {
    other_bfc_id = 1;
  }
  if (bfc_status_vec[bfc_id].CondForIsolatedProfiling()) {
    bfc_status_vec[bfc_id].IsolatedProfiling(total_region_allocated_bytes_, alloc_dealloc);
  }
  if (bfc_status_vec[bfc_id].CondForIsolatedProfiling_2()) {
  // if (bfc_status_vec[bfc_id].current_step_id > start_aligning_step + num_isolated_profiling) {
    bfc_status_vec[bfc_id].IsolatedProfiling_2(total_region_allocated_bytes_, alloc_dealloc);
  }
  if (use_colocated_simulator) {
    if (bfc_status_vec[bfc_id].CondForColocatedProfiling()) {
      bfc_status_vec[bfc_id].ColocatedProfiling(total_region_allocated_bytes_, alloc_dealloc);
    }
  }
}

} // namespace tensorflow
