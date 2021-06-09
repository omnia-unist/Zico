#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COORDINATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COORDINATOR_H_

#include <chrono>
#include <ctime>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <thread>

#include "stdio.h"

#include "tensorflow/core/common_runtime/bfc_allocator.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/core/common_runtime/gangmuk_parser.h"
// #include "tensorflow/core/common_runtime/global_pool.h"

namespace tensorflow {

typedef std::chrono::time_point<
    std::chrono::_V2::steady_clock,
    std::chrono::duration<long int, std::ratio<1, 1000000000> > >
    time_point;
    
class Device;

class Coordinator {
  public:
    std::string name = "Coordinator";
    std::string GPU_0_allocator_name = "GPU_0_bfc";
    std::string GPU_1_allocator_name = "GPU_1_bfc";
    std::string GPU_2_allocator_name = "GPU_2_bfc";
    std::string GPU_3_allocator_name = "GPU_3_bfc";

    int num_executor;
    int current_step_id = -9999;
    // GM_Parser parser; => 이거 안됨... 컴파일에러.. 이유는 아직 못했음. 선언 및 정의의 순서 문제 인듯.
    int start_alinging_step;
    int gpu_0_current_step_id = -9999;
    int gpu_1_current_step_id = -9999;
    int gpu_2_current_step_id = -9999;
    int gpu_3_current_step_id = -9999;

    bool gpu_0_backward_passed = false;
    bool gpu_1_backward_passed = false;
    bool gpu_2_backward_passed = false;
    bool gpu_3_backward_passed = false;

    bool gpu_1_is_first_node = false;
    bool gpu_3_is_first_node = false;

    volatile bool gpu_0_synced = false;
    volatile bool gpu_2_synced = false;
    volatile bool gpu_1_is_waiting = false;
    volatile bool gpu_3_is_waiting = false;

    volatile bool all_ready = false;
    volatile int num_ready_gpu = 0;

    volatile bool initialized = false;
    std::string coordinator_policy = "not_init";

    long long gpu_0_peak_memory = 0;
    long long gpu_1_peak_memory = 0;
    long long gpu_2_peak_memory = 0;
    long long gpu_3_peak_memory = 0;

    long long gpu_0_p95_memory = 0;
    long long gpu_1_p95_memory = 0;
    long long gpu_2_p95_memory = 0;
    long long gpu_3_p95_memory = 0;

    unsigned long long executor_iteration_time = 0;
    unsigned long long executor_iteration_start_time = 0;

    volatile unsigned long long gpu_0_cur_execution_id = 0;
    volatile unsigned long long gpu_1_cur_execution_id = 0;
    volatile unsigned long long gpu_0_last_execution_id = 0;
    volatile unsigned long long gpu_1_last_execution_id = 0;
    volatile bool gpu_0_end_of_step = false;
    volatile bool gpu_1_end_of_step = false;

    unsigned long long gpu_0_peak_execution_id = 0;
    unsigned long long gpu_1_peak_execution_id = 0;
    unsigned long long gpu_0_temp_peak_execution_id = 0;
    unsigned long long gpu_1_temp_peak_execution_id = 0;
    long long gpu_0_temp_peak_bytes = 0;
    long long gpu_1_temp_peak_bytes = 0;
    long long gpu_0_peak_bytes = 0;
    long long gpu_1_peak_bytes = 0;

   public:
    Coordinator();
    ~Coordinator();
    
    void Initialization(Device* device) {
      coordinator_policy = device->CoordinatorPolicy();
      start_alinging_step = device->StartAligningStep();
      printf("CEvent Coordinator Policy: %s\n", coordinator_policy.c_str());
      fflush(stdout);
      for (int i = 0; i < 4; i++) {
        step_id_vec.push_back(-9999); // hard coding, 하드 코딩
      }
      initialized = true;
    }
    
    int BFCId(std::string allocator_name) {
      if (allocator_name == GPU_0_allocator_name)
        return 0; 
      else if (allocator_name == GPU_1_allocator_name)
        return 1; 
      else if (allocator_name == GPU_2_allocator_name)
        return 2; 
      else if (allocator_name == GPU_3_allocator_name)
        return 3;
      else {
        printf("GANGMUK_ERROR: %s is not supported gpu name by Coordinator\n", allocator_name.c_str());
        fflush(stdout);
        printf("%s\n", GPU_0_allocator_name.c_str());
        fflush(stdout);
        printf("%s\n", GPU_1_allocator_name.c_str());
        fflush(stdout);
        printf("%s\n", GPU_2_allocator_name.c_str());
        printf("%s\n", GPU_3_allocator_name.c_str());
        fflush(stdout);
        CHECK_EQ(0,1);
      }
    }

    bool Policy_0(long long step_id, int node_id, std::string node_name, Device* device) {
      Allocator* allocator_ptr = device->GetAllocator(AllocatorAttributes());
      std::string allocator_name = allocator_ptr->Name();
      if (allocator_name == "GPU_0_bfc") {
        if (gpu_0_current_step_id != step_id && (step_id == start_alinging_step || step_id == start_alinging_step + 1)) {
          printf("%s,NEW_STEP,%d\n", allocator_name.c_str(), step_id);
          gpu_0_peak_memory = BFCAllocator::cr_global_pool_.bfc_status_vec[0].chunk_peak_bytes;
          gpu_0_p95_memory = gpu_0_peak_memory * 0.95;
          printf("(%s) CEvent PeakMemory: %lld, GPU_0_P95_memory: %lld MB\n",
                 allocator_name.c_str(), gpu_0_peak_memory / (1024 * 1024),
                 gpu_0_p95_memory / (1024 * 1024));
          gpu_0_current_step_id = step_id;
          gpu_0_backward_passed = false;
        }
        if (gpu_0_backward_passed == false) {
          long long gpu_0_cur_bytes =
          BFCAllocator::cr_global_pool_.bfc_status_vec[0].chunk_bytes;
          if (gpu_0_cur_bytes > gpu_0_p95_memory) {
            bool print_once_1 = true;
            bool print_once_2 = true;
            while(true) {
              if (print_once_1) {
                printf("%s,%d,CEvent Reach to Backward Start with P95,(current_chunk_bytes,%lld MB, P95,%lld MB, PeakMemory,%lld MB (node_id: %d) \n", allocator_name.c_str(), step_id, gpu_0_cur_bytes/(1024*1024), gpu_0_p95_memory/(1024*1024), gpu_0_peak_memory/ (1204 * 1024), node_id);
                fflush(stdout);
                print_once_1 = false;
              }
              if (gpu_1_is_waiting) {
                print_current_state(allocator_name, "CEvent Send Sync to GPU_1", step_id, node_id, "-");
                gpu_0_synced = true;
                gpu_0_backward_passed = true;
                break;
              }
              else {
                if (print_once_2) {
                  printf("(%s) CEvent Wait For GPU_1 (while loop), step_id: %d,(gpu_0_cur_bytes: %lld, P95: %lld), (%s) \n",
                          allocator_name.c_str(), step_id, gpu_0_cur_bytes/(1024*1024), gpu_0_p95_memory/(1024*1024), node_name.c_str());
                  fflush(stdout);
                  print_once_2 = false;
                }
              }
            }
          }
        }
        if (device->PrintCoordinator() > 2) {
          print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
        }
        return true;
      }

      else if (allocator_name == "GPU_1_bfc") {
        std::string GPU_1_forward_start_node_name = BFCAllocator::cr_global_pool_.bfc_status_vec[1].forward_start_node_name;
        if (gpu_1_current_step_id != step_id && (step_id == start_alinging_step || step_id == start_alinging_step + 1) && node_name.compare(GPU_1_forward_start_node_name) == 0) {
          printf("%s,NEW_STEP,%d\n", allocator_name.c_str(), step_id);
          gpu_1_current_step_id = step_id;
          CHECK_EQ(gpu_1_is_waiting, false);
          bool print_once_1 = true;
          bool print_once_2 = true;
          gpu_1_is_waiting = true;
          while (true) {
            if (print_once_1) {
              printf("(%s) CEvent Reach to Forward Start, step_id: %d (%s) \n", 
                      allocator_name.c_str(), step_id, node_name.c_str());
              fflush(stdout);
              print_once_1 = false;
            }
            if (gpu_0_synced) {
              print_current_state(allocator_name, "CEvent Receive Sync", step_id, node_id, node_name);
              gpu_0_synced = false;
              gpu_1_is_waiting = false;
              break;
            }
            else {
              // GPU_0_bfc이 백워드 시작 노드에 도달하지 못했으면 기다린다.
              gpu_1_is_waiting = true; // 불필요해보이는데 일단 추가.
              if (print_once_2) {
                print_current_state(allocator_name, "CEvent Wait For GPU_0", step_id, node_id, node_name);
                print_once_2 = false;
              }
            }
          }
          return true;
        }
        // 포워드 스타드가 아닌 노드들은 실행 허가
        else {
          if (device->PrintCoordinator() > 2) {
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
          return true;
        }
      }
    }

    bool Policy_1(long long step_id, int node_id, std::string node_name, Device* device) {
      Allocator* allocator_ptr = device->GetAllocator(AllocatorAttributes());
      std::string allocator_name = allocator_ptr->Name();
      std::string GPU_1_forward_start_node_name = BFCAllocator::cr_global_pool_.bfc_status_vec[1].forward_start_node_name;
      std::string GPU_0_backward_start_node_name = BFCAllocator::cr_global_pool_.bfc_status_vec[0].backward_start_node_name;
      if (gpu_0_current_step_id != step_id && step_id > gpu_0_current_step_id && allocator_name == "GPU_0_bfc") {
        gpu_0_current_step_id = step_id;
        gpu_0_backward_passed = false;
      }

      if (gpu_1_current_step_id != step_id && step_id > gpu_1_current_step_id && allocator_name == "GPU_1_bfc" && node_name.compare(GPU_1_forward_start_node_name) == 0) {
        gpu_1_current_step_id = step_id;
        gpu_1_is_first_node = true;
      }

      if (allocator_name == "GPU_0_bfc") {
        if (node_name.compare(GPU_0_backward_start_node_name) == 0 && gpu_0_backward_passed == false) {
          long long ninety_percentile = BFCAllocator::cr_global_pool_.bfc_status_vec[0].chunk_peak_bytes * 0.9;
          long long gpu_0_cur_bytes = BFCAllocator::cr_global_pool_.bfc_status_vec[0].chunk_bytes;
          if (gpu_0_cur_bytes > ninety_percentile) {
            bool print_once_1 = true;
            bool print_once_2 = true;
            while(true) {
              if (print_once_1) {
                printf("(%s) CEvent Reach to Backward Start with P90, step_id: %d, (gpu_0_cur_bytes: %lld,P90: %lld), (%s) \n", 
                        allocator_name.c_str(), step_id, gpu_0_cur_bytes/(1024*1024), ninety_percentile/(1024*1024), node_name.c_str());
                fflush(stdout);
                print_once_1 = false;
              }
              if (gpu_1_is_waiting) {
                print_current_state(allocator_name, "CEvent Send Sync to awaiting GPU_0", step_id, node_id, "-");
                gpu_0_synced = true;
                gpu_0_backward_passed = true;
                break;
              }
              else {
                if (print_once_2) {
                  printf("(%s) CEvent Wait For GPU_1, step_id: %d, (gpu_0_cur_bytes: %lld, P90: %lld), (%s) \n", 
                          allocator_name.c_str(), step_id, gpu_0_cur_bytes/(1024*1024), ninety_percentile/(1024*1024), node_name.c_str());
                  fflush(stdout);
                  print_once_2 = false;
                }
              }
            }
          }
          else {
            printf("(%s) CEvent Backward Start but No P90, step_id: %d, (gpu_0_cur_bytes: %lld, P90: %lld), (%s)\n", 
                    allocator_name.c_str(), step_id, gpu_0_cur_bytes/(1024*1024), ninety_percentile/(1024*1024), node_name.c_str());
            fflush(stdout);
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
          return true;
        }
        else {
          if (device->PrintCoordinator() > 2) {
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
          return true;
        }
      }

      // forward start node
      else if (allocator_name == "GPU_1_bfc") {
        if (node_name.compare(GPU_1_forward_start_node_name) == 0 && gpu_1_is_first_node) {
          CHECK_EQ(gpu_1_is_waiting, false);
          bool print_once_1 = true;
          bool print_once_2 = true;
          gpu_1_is_waiting = true;
          while (true) {
            if (print_once_1) {
              printf("(%s) CEvent Reach to Forward Start, step_id: %d (%s) \n", 
                      allocator_name.c_str(), step_id, node_name.c_str());
              fflush(stdout);
              print_once_1 = false;
            }
            if (gpu_0_synced) { // 이게 업데이트가 안된다. GPU_0에 의해서 true로 바뀌어야하는데.
              print_current_state(allocator_name, "CEvent Receive Sync", step_id, node_id, node_name);
              gpu_0_synced = false;
              gpu_1_is_waiting = false;
              gpu_1_is_first_node = false;
              break;
            }
            else {
              // GPU_0_bfc이 백워드 시작 노드에 도달하지 못했으면 기다린다.
              gpu_1_is_waiting = true; // 불필요해보이는데 일단 추가.
              if (print_once_2) {
                print_current_state(allocator_name, "CEvent Wait For GPU_0", step_id, node_id, node_name);
                print_once_2 = false;
              }
            }
          }
          return true;
        }
        // 포워드 스타드가 아닌 노드들은 실행 허가
        else {
          if (device->PrintCoordinator() > 2) {
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
          return true;
        }
      }
    }

    bool Policy_2(long long step_id, int node_id, std::string node_name, Device* device) {
      Allocator* allocator_ptr = device->GetAllocator(AllocatorAttributes());
      std::string allocator_name = allocator_ptr->Name();
      if (allocator_name == "GPU_0_bfc") {
        if (gpu_0_current_step_id != step_id && step_id > gpu_0_current_step_id) {
          printf("%s,NEW_STEP,%d\n", allocator_name.c_str(), step_id);
          gpu_0_peak_memory = BFCAllocator::cr_global_pool_.bfc_status_vec[0].chunk_peak_bytes;
          gpu_0_p95_memory = gpu_0_peak_memory * 0.95;
          printf("(%s) CEvent PeakMemory: %lld, GPU_0_P95_memory: %lld MB\n", 
                  allocator_name.c_str(), 
                  gpu_0_peak_memory/(1024*1024), 
                  gpu_0_p95_memory/(1024*1024));
          gpu_0_current_step_id = step_id;
          gpu_0_backward_passed = false;
        }
      }

      if (allocator_name == "GPU_0_bfc") {
        if (gpu_0_backward_passed == false) {
          long long gpu_0_cur_bytes = BFCAllocator::cr_global_pool_.bfc_status_vec[0].chunk_bytes;
          if (gpu_0_cur_bytes > gpu_0_p95_memory) {
            bool print_once_1 = true;
            bool print_once_2 = true;
            while(true) {
              if (print_once_1) {
                printf("%s,%d,CEvent Reach to Backward Start with P95, (current_chunk_bytes,%lld MB, "
                      "P95,%lld MB, PeakMemory,%lld MB (node_id: %d) \n", 
                        allocator_name.c_str(), step_id, gpu_0_cur_bytes/(1024*1024), 
                        gpu_0_p95_memory/(1024*1024), gpu_0_peak_memory/ (1204 * 1024), 
                        node_id);
                fflush(stdout);
                print_once_1 = false;
              }
              if (gpu_1_is_waiting) {
                print_current_state(allocator_name, "CEvent Send Sync to GPU_1", step_id, node_id, "-");
                gpu_0_synced = true;
                gpu_0_backward_passed = true;
                break;
              }
              else {
                if (print_once_2) {
                  printf("(%s) CEvent Wait For GPU_1 (while loop), step_id: %d, (gpu_0_cur_bytes: %lld, P95: %lld), (%s) \n", 
                          allocator_name.c_str(), step_id, gpu_0_cur_bytes/(1024*1024), gpu_0_p95_memory/(1024*1024), 
                          node_name.c_str());
                  fflush(stdout);
                  print_once_2 = false;
                }
              }
            }
          }
        }
        if (device->PrintCoordinator() > 2) {
          print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
        }
        return true;
      }

      else if (allocator_name == "GPU_1_bfc") {
        std::string GPU_1_forward_start_node_name = BFCAllocator::cr_global_pool_.bfc_status_vec[1].forward_start_node_name;
        if (gpu_1_current_step_id != step_id && 
            step_id > gpu_1_current_step_id &&
            node_name.compare(GPU_1_forward_start_node_name) == 0) {
          gpu_1_current_step_id = step_id;
          CHECK_EQ(gpu_1_is_waiting, false);
          bool print_once_1 = true;
          bool print_once_2 = true;
          gpu_1_is_waiting = true;
          while (true) {
            if (print_once_1) {
              printf("(%s) CEvent Reach to Forward Start, step_id: %d (%s) \n", 
                      allocator_name.c_str(), step_id, node_name.c_str());
              fflush(stdout);
              print_once_1 = false;
            }
            if (gpu_0_synced) {
              print_current_state(allocator_name, "CEvent Receive Sync", step_id, node_id, node_name);
              gpu_0_synced = false;
              gpu_1_is_waiting = false;
              break;
            }
            else {
              // GPU_0_bfc이 백워드 시작 노드에 도달하지 못했으면 기다린다.
              gpu_1_is_waiting = true; // 불필요해보이는데 일단 추가.
              if (print_once_2) {
                print_current_state(allocator_name, "CEvent Wait For GPU_0", step_id, node_id, node_name);
                print_once_2 = false;
              }
            }
          }
          return true;
        }
        // 포워드 스타드가 아닌 노드들은 실행 허가
        else {
          if (device->PrintCoordinator() > 2) {
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
          return true;
        }
      }
    }

    bool Policy_3(long long step_id, int node_id, std::string node_name, Device* device) {
      Allocator* allocator_ptr = device->GetAllocator(AllocatorAttributes());
      std::string allocator_name = allocator_ptr->Name();
      if (allocator_name == "GPU_0_bfc") {
        if (gpu_0_current_step_id != step_id && step_id > gpu_0_current_step_id) {
          printf("%s,NEW_STEP,%d\n", allocator_name.c_str(), step_id);
          gpu_0_current_step_id = step_id;
          if (device->PrintCoordinator() > 2) {
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
        }
      }
      if (allocator_name == "GPU_1_bfc") {
        std::string GPU_1_forward_start_node_name = BFCAllocator::cr_global_pool_.bfc_status_vec[1].forward_start_node_name;
        if (gpu_1_current_step_id != step_id && 
            step_id > gpu_1_current_step_id &&
            node_name.compare(GPU_1_forward_start_node_name) == 0) {
          printf("%s,NEW_STEP,%d\n", allocator_name.c_str(), step_id);
          gpu_1_current_step_id = step_id;
          bool print_once = true;
          while (true) {
            if (print_once) {
              print_current_state(allocator_name,"CEvent Wait for GPU_0", step_id, node_id, node_name);
              print_once = false;
            }
            //if (BFCAllocator::cr_global_pool_.bfc_status_vec[0].reach_to_last_alloc) { // last allocation based
            if (BFCAllocator::cr_global_pool_.bfc_status_vec[0].reach_to_last_dealloc) { // last deallocation based
              print_current_state(allocator_name,
                                  "CEvent Detect last allocation of GPU_0",
                                  step_id, node_id, node_name);
              gpu_1_is_first_node = false;
              break;
            }
          }
          if (device->PrintCoordinator() > 2) {
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
          return true;
        } else {
          // Unnecessary else statement. Just for better readability.
          if (device->PrintCoordinator() > 2) {
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
          return true;
        }
      }
    }

    bool Policy_4(long long step_id, int node_id, std::string node_name, Device* device) {
      Allocator* allocator_ptr = device->GetAllocator(AllocatorAttributes());
      std::string allocator_name = allocator_ptr->Name();
      if (allocator_name == "GPU_0_bfc") {
        ++gpu_0_cur_execution_id;
        if (gpu_0_current_step_id != step_id && step_id > gpu_0_current_step_id) {
          BFCAllocator::cr_global_pool_.bfc_status_vec[0].is_forward = true; // 새로운 스텝이 시작되면 포워드 패스가 시작된거임.
          gpu_0_peak_memory =
              BFCAllocator::cr_global_pool_.bfc_status_vec[0].chunk_peak_bytes;
          gpu_0_peak_execution_id = gpu_0_temp_peak_execution_id;
          gpu_0_peak_bytes = gpu_0_temp_peak_bytes;
          gpu_0_last_execution_id = gpu_0_cur_execution_id;
          printf(
              "(%s) "
              "step_id,%d, gpu_0_peak_execution_id,%llu, gpu_0_peak_bytes,%lld, "
              "gpu_0_peak_memory,%lld, gpu_0_last_execution_id,%llu\n",
              allocator_name.c_str(), step_id, gpu_0_peak_execution_id,
              gpu_0_peak_bytes / (1024 * 1024), gpu_0_peak_memory / (1024 * 1024),
              gpu_0_last_execution_id);
          gpu_0_cur_execution_id = 0;
          gpu_0_temp_peak_execution_id = 0;
          gpu_0_temp_peak_bytes = 0;
          gpu_0_backward_passed = false;
          gpu_0_current_step_id = step_id;
        }
        if (gpu_0_backward_passed == false) {
          long long gpu_0_cur_bytes = BFCAllocator::cr_global_pool_.bfc_status_vec[0].chunk_bytes;
          long long gpu_0_cur_region_bytes = BFCAllocator::cr_global_pool_.bfc_status_vec[0].region_bytes;
          if (gpu_0_cur_bytes > gpu_0_temp_peak_bytes &&
              gpu_0_cur_execution_id != 0) {
            // if (gpu_0_cur_region_bytes > gpu_0_temp_peak_bytes) {
            gpu_0_temp_peak_bytes = gpu_0_cur_bytes;
            gpu_0_temp_peak_execution_id = gpu_0_cur_execution_id;
          }
          if (gpu_0_peak_execution_id != 0 && gpu_0_cur_execution_id == gpu_0_peak_execution_id) {
            BFCAllocator::cr_global_pool_.bfc_status_vec[0].is_forward = false;
            bool print_once = true;
            while(true) {
              if (print_once) {
                printf("(%s) (step_id,%d) CEvent Reach to peak_execution_id,%llu (gpu_0_cur_bytes: %lld)\n", 
                        allocator_name.c_str(), step_id, gpu_0_cur_execution_id, gpu_0_cur_bytes/(1024*1024));
                fflush(stdout);
                print_once = false;
              }
              if (gpu_1_is_waiting) {
                print_current_state(allocator_name, "CEvent Send Sync to GPU_1", step_id, node_id, "-");
                gpu_0_synced = true;
                gpu_0_backward_passed = true;
                break;
              }
            }
          }
        }
        if (device->PrintCoordinator() > 2) {
          print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
        }
        return true;
      }

      else if (allocator_name == "GPU_1_bfc") {
        ++gpu_1_cur_execution_id;
        std::string GPU_1_forward_start_node_name = BFCAllocator::cr_global_pool_.bfc_status_vec[1].forward_start_node_name;
        if (gpu_1_current_step_id != step_id && step_id > gpu_1_current_step_id && node_name.compare(GPU_1_forward_start_node_name) == 0) {
          BFCAllocator::cr_global_pool_.bfc_status_vec[1].is_forward = true;
          gpu_1_peak_memory = BFCAllocator::cr_global_pool_.bfc_status_vec[1].chunk_peak_bytes;
          gpu_1_peak_execution_id = gpu_1_temp_peak_execution_id;
          gpu_1_peak_bytes = gpu_1_temp_peak_bytes;
          gpu_1_last_execution_id = gpu_1_cur_execution_id;
          printf("(%s) step_id,%d, gpu_1_peak_execution_id,%llu, gpu_1_peak_bytes,%lld, gpu_1_peak_memory,%lld, gpu_1_last_execution_id,%llu\n",
              allocator_name.c_str(), step_id, gpu_1_peak_execution_id, gpu_1_peak_bytes / (1024 * 1024), gpu_1_peak_memory / (1024 * 1024), gpu_1_last_execution_id);
          gpu_1_cur_execution_id = 0;
          gpu_1_temp_peak_execution_id = 0;
          gpu_1_temp_peak_bytes = 0;
          gpu_1_backward_passed = false;
          gpu_1_current_step_id = step_id;
          CHECK_EQ(gpu_1_is_waiting, false);
          bool print_once_1 = true;
          bool print_once_2 = true;
          gpu_1_is_waiting = true;
          while (true) {
            if (print_once_1) {
              printf("(%s) CEvent Reach to Forward Start, step_id: %d (%s) \n", allocator_name.c_str(), step_id, node_name.c_str());
              fflush(stdout);
              print_once_1 = false;
            }
            if (gpu_0_synced) {
              print_current_state(allocator_name, "CEvent Receive Sync", step_id, node_id, node_name);
              gpu_0_synced = false;
              gpu_1_is_waiting = false;
              break;
            }
          }
          return true;
        }
        // if (gpu_1_backward_passed == false) {
          long long gpu_1_cur_bytes = BFCAllocator::cr_global_pool_.bfc_status_vec[1].chunk_bytes;
          if (gpu_1_cur_bytes > gpu_1_temp_peak_bytes && gpu_1_cur_execution_id != 0) {
            gpu_1_temp_peak_bytes = gpu_1_cur_bytes;
            gpu_1_temp_peak_execution_id = gpu_1_cur_execution_id;
          }
          if (gpu_1_cur_execution_id == gpu_1_peak_execution_id) { // Backward detection condition
            BFCAllocator::cr_global_pool_.bfc_status_vec[1].is_forward = false;
            // gpu_1_backward_passed = true;
          }
        // }
        if (device->PrintCoordinator() > 2) {
          print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
        }
        return true;
      }
    }

    bool Policy_5(long long step_id, int node_id, std::string node_name, Device* device) {
      Allocator* allocator_ptr = device->GetAllocator(AllocatorAttributes());
      std::string allocator_name = allocator_ptr->Name();
      if (allocator_name == "GPU_0_bfc" || allocator_name == "GPU_1_bfc") {
        if (allocator_name == "GPU_0_bfc") {
          if (gpu_0_current_step_id != step_id && step_id > gpu_0_current_step_id) {
            printf("%s,NEW_STEP,%d\n", allocator_name.c_str(), step_id);
            gpu_0_peak_memory = BFCAllocator::cr_global_pool_.bfc_status_vec[0].chunk_peak_bytes;
            gpu_0_p95_memory = gpu_0_peak_memory * 0.95;
            printf("(%s) CEvent Previous step peak memory: %lld MB, GPU_0_P95_memory: %lld MB\n", 
                    allocator_name.c_str(), gpu_0_peak_memory/(1024*1024), gpu_0_p95_memory/(1024*1024));
            gpu_0_current_step_id = step_id;
            gpu_0_backward_passed = false;
          }
          if (gpu_0_backward_passed == false) {
            long long gpu_0_cur_bytes = BFCAllocator::cr_global_pool_.bfc_status_vec[0].chunk_bytes;
            if (gpu_0_cur_bytes > gpu_0_p95_memory) {
              bool print_once_1 = true;
              bool print_once_2 = true;
              while(true) {
                if (print_once_1) {
                  printf("(%s) CEvent Reach to Backward Start with P95, step_id: %d, (gpu_0_cur_bytes: %lld MB > P95: %lld MB), (node: %s) \n", 
                          allocator_name.c_str(), step_id, gpu_0_cur_bytes/(1024*1024), gpu_0_p95_memory/(1024*1024), node_name.c_str());
                  fflush(stdout);
                  print_once_1 = false;
                }
                if (gpu_1_is_waiting) {
                  print_current_state(allocator_name, "CEvent Send Sync to awaiting GPU_0", step_id, node_id, "-");
                  gpu_0_synced = true;
                  gpu_0_backward_passed = true;
                  break;
                }
                else {
                  if (print_once_2) {
                    printf("(%s) CEvent Wait For GPU_1 (while loop), step_id: %d, (gpu_0_cur_bytes: %lld, P95: %lld), (%s) \n", 
                            allocator_name.c_str(), step_id, gpu_0_cur_bytes/(1024*1024), gpu_0_p95_memory/(1024*1024), node_name.c_str());
                    fflush(stdout);
                    print_once_2 = false;
                  }
                }
              }
            }
          }
          if (device->PrintCoordinator() > 2) {
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
          return true;
        }

        // forward start node
        else if (allocator_name == "GPU_1_bfc") {
          std::string GPU_1_forward_start_node_name = BFCAllocator::cr_global_pool_.bfc_status_vec[1].forward_start_node_name;
          if (gpu_1_current_step_id != step_id && step_id > gpu_1_current_step_id && node_name.compare(GPU_1_forward_start_node_name) == 0) {
            gpu_1_current_step_id = step_id;
            gpu_1_is_first_node = true;
          }
          if (node_name.compare(GPU_1_forward_start_node_name) == 0 && gpu_1_is_first_node) {
            CHECK_EQ(gpu_1_is_waiting, false);
            bool print_once_1 = true;
            bool print_once_2 = true;
            gpu_1_is_waiting = true;
            while (true) {
              if (print_once_1) {
                printf("(%s) CEvent Reach to Forward Start, step_id: %d (%s) \n", 
                        allocator_name.c_str(), step_id, node_name.c_str());
                fflush(stdout);
                print_once_1 = false;
              }
              if (gpu_0_synced) { // 이게 업데이트가 안된다. GPU_0에 의해서 true로 바뀌어야하는데.
                print_current_state(allocator_name, "CEvent Receive Sync", step_id, node_id, node_name);
                gpu_0_synced = false;
                gpu_1_is_waiting = false;
                gpu_1_is_first_node = false;
                break;
              }
              else {
                // GPU_0_bfc이 백워드 시작 노드에 도달하지 못했으면 기다린다.
                gpu_1_is_waiting = true; // 불필요해보이는데 일단 추가.
                if (print_once_2) {
                  print_current_state(allocator_name, "CEvent Wait For GPU_0", step_id, node_id, node_name);
                  print_once_2 = false;
                }
              }
            }
            return true;
          }
          // 포워드 스타드가 아닌 노드들은 실행 허가
          else {
            if (device->PrintCoordinator() > 2) {
              print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
            }
            return true;
          }
        }
      }

      if (allocator_name == "GPU_2_bfc" || allocator_name == "GPU_3_bfc") {
        if (allocator_name == "GPU_2_bfc") {
          if (gpu_2_current_step_id != step_id &&
              step_id > gpu_2_current_step_id) {
            printf("%s,NEW_STEP,%d\n", allocator_name.c_str(), step_id);
            gpu_2_peak_memory = BFCAllocator::cr_global_pool_.bfc_status_vec[2].chunk_peak_bytes;
            gpu_2_p95_memory = gpu_2_peak_memory * 0.95;
            printf("(%s) CEvent Previous step peak memory: %lld MB, GPU_2_P95_memory: %lld MB\n", 
                    allocator_name.c_str(), gpu_2_peak_memory/(1024*1024), gpu_2_p95_memory/(1024*1024));
            gpu_2_current_step_id = step_id;
            gpu_2_backward_passed = false;
          }
          if (gpu_2_backward_passed == false) {
            long long gpu_2_cur_bytes = BFCAllocator::cr_global_pool_.bfc_status_vec[2].chunk_bytes;
            if (gpu_2_cur_bytes > gpu_2_p95_memory) {
              bool print_once_1 = true;
              bool print_once_2 = true;
              while(true) {
                if (print_once_1) {
                  printf("(%s) CEvent Reach to Backward Start with P95, step_id: %d, (gpu_2_cur_bytes: %lld MB > P95: %lld MB), (%s) \n", 
                         allocator_name.c_str(), step_id, gpu_2_cur_bytes/(1024*1024), gpu_2_p95_memory/(1024*1024), node_name.c_str());
                  fflush(stdout);
                  print_once_1 = false;
                }
                if (gpu_3_is_waiting) {
                  print_current_state(allocator_name, "CEvent Send Sync to awaiting GPU_2", step_id, node_id, "-");
                  gpu_2_synced = true;
                  gpu_2_backward_passed = true;
                  break;
                }
                else {
                  if (print_once_2) {
                    printf("(%s) CEvent Wait For GPU_3 (while loop), step_id: %d, (gpu_2_cur_bytes: %lld, P95: %lld), (%s) \n", 
                            allocator_name.c_str(), step_id, gpu_2_cur_bytes/(1024*1024), gpu_2_p95_memory/(1024*1024), node_name.c_str());
                    fflush(stdout);
                    print_once_2 = false;
                  }
                }
              }
            }
          }
          if (device->PrintCoordinator() > 2) {
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
          return true;
        }

        // forward start node
        else if (allocator_name == "GPU_3_bfc") {
          // std::string GPU_3_forward_start_node_name = BFCAllocator::cr_global_pool_.bfc_status_vec[3].forward_start_node_name;
          // std::string GPU_3_forward_start_node_name = forward_start_node_name;
          std::string GPU_3_forward_start_node_name = "";
          if (gpu_3_current_step_id != step_id && step_id > gpu_3_current_step_id && node_name.compare(GPU_3_forward_start_node_name) == 0) {
            gpu_3_current_step_id = step_id;
            gpu_3_is_first_node = true;
          }
          if (node_name.compare(GPU_3_forward_start_node_name) == 0 && gpu_3_is_first_node) {
            CHECK_EQ(gpu_3_is_waiting, false);
            bool print_once_1 = true;
            bool print_once_2 = true;
            gpu_3_is_waiting = true;
            while (true) {
              if (print_once_1) {
                printf("(%s) CEvent Reach to Forward Start, step_id: %d (%s) \n", allocator_name.c_str(), step_id, node_name.c_str());
                fflush(stdout);
                print_once_1 = false;
              }
              if (gpu_2_synced) { // 이게 업데이트가 안된다. GPU_0에 의해서 true로 바뀌어야하는데.
                print_current_state(allocator_name, "CEvent Receive Sync", step_id, node_id, node_name);
                gpu_2_synced = false;
                gpu_3_is_waiting = false;
                gpu_3_is_first_node = false;
                break;
              }
              else {
                // GPU_0_bfc이 백워드 시작 노드에 도달하지 못했으면 기다린다.
                gpu_3_is_waiting = true; // 불필요해보이는데 일단 추가.
                if (print_once_2) {
                  print_current_state(allocator_name, "CEvent Wait For GPU_2", step_id, node_id, node_name);
                  print_once_2 = false;
                }
              }
            }
            return true;
          }
          // 포워드 스타드가 아닌 노드들은 실행 허가
          else {
            if (device->PrintCoordinator() > 2) {
              print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
            }
            return true;
          }
        }
      }
    }

    bool Policy_6(long long step_id, int node_id, std::string node_name, Device* device) {
      Allocator* allocator_ptr = device->GetAllocator(AllocatorAttributes());
      std::string allocator_name = allocator_ptr->Name();
      if (allocator_name == "GPU_0_bfc") {
        ++gpu_0_cur_execution_id;
        if (gpu_0_current_step_id != step_id && step_id > gpu_0_current_step_id) {
          gpu_0_current_step_id = step_id;
          gpu_0_last_execution_id = gpu_0_cur_execution_id;
          printf("%s,NEW_STEP,%d,ExecutorIterTime,%llu,gpu_0_last_execution_id,%llu\n",
                 allocator_name.c_str(), step_id, executor_iteration_time, gpu_0_last_execution_id);
          executor_iteration_start_time = Env::Default()->NowMicros();
          gpu_0_cur_execution_id = 0;
          if (device->PrintCoordinator() > 2) {
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
        }
        if (gpu_0_cur_execution_id == gpu_0_last_execution_id) {
          executor_iteration_time = Env::Default()->NowMicros() - executor_iteration_start_time;
          gpu_0_synced = true;
        }
      }
      if (allocator_name == "GPU_1_bfc") {
        ++gpu_1_cur_execution_id;
        if (gpu_1_current_step_id != step_id && step_id > gpu_1_current_step_id) {
          printf("%s,NEW_STEP,%d\n", allocator_name.c_str(), step_id);
          gpu_1_current_step_id = step_id;
          gpu_1_last_execution_id = gpu_1_cur_execution_id;
          gpu_1_cur_execution_id = 0;
          bool print_once = true;
          while (true) {
            if (print_once) {
              print_current_state(allocator_name,"CEvent Wait for GPU_0", step_id, node_id, node_name);
              print_once = false;
            }
            if (gpu_0_synced) { // last deallocation based
              print_current_state(allocator_name,"CEvent Detect GPU_0 last_execution",step_id, node_id, node_name);
              gpu_0_synced = false;
              break;
            }
          }
          if (device->PrintCoordinator() > 2) {
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
          return true;
        } else {
          // Unnecessary else statement. Just for better readability.
          if (device->PrintCoordinator() > 2) {
            print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
          }
          return true;
        }
      }
    }

    std::vector<int> step_id_vec;

    bool IsNewStep(int bfc_id, int step_id) {
      if (step_id_vec[bfc_id] != step_id) {
        step_id_vec[bfc_id] = step_id;
        return true;
      }
      return false;
    }

    bool Policy_8(long long step_id, int node_id, std::string node_name, Device* device) {
      Allocator* allocator_ptr = device->GetAllocator(AllocatorAttributes());
      std::string allocator_name = allocator_ptr->Name();
      // BFCAllocator* bfc_ptr = reinterpret_cast<BFCAllocator*>(allocator_ptr); //컴파일 성공..!?
      // bfc_ptr->foo();
      int bfc_id = allocator_ptr->BFCId();
      if (device->PrintCoordinator() > 2) {
        print_current_state(allocator_name, "Processing", step_id, node_id, node_name);
      }
      if (allocator_ptr->IsNewStep(step_id)) {
      // if (IsNewStep(bfc_id, step_id)) {
        allocator_ptr->UpdateNewStep(step_id);
        BFCAllocator::cr_global_pool_.bfc_status_vec[bfc_id].EndOfOldStep(current_step_id);
        print_current_state(allocator_name, "Executor,NewStep", step_id, node_id, node_name);
        BFCAllocator::cr_global_pool_.SimulateIfNewStep(bfc_id, step_id);
        BFCAllocator::cr_global_pool_.bfc_status_vec[bfc_id].ActualStartOfNewStep();
      }
    }

    bool Coordination(long long step_id, int node_id, std::string node_name, Device* device);

    void print_current_state(std::string allocator_name, std::string cur_state, long long step_id, int node_id, std::string node_name);

    bool AskToGlobalPool(int node_id, std::string node_name, Device* device);

    bool PermitNodeExecution(int node_id, std::string node_name, Device* device);

};

class CoordinatorWrapper {
  public:
    static Coordinator coordinator_;
};

}  // namespace tensorflow

#endif