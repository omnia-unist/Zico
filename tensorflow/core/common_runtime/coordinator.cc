#include "tensorflow/core/common_runtime/coordinator.h"

namespace tensorflow {
  
Coordinator CoordinatorWrapper::coordinator_;

// multiple definition error
// GlobalPool BFCAllocator::cr_global_pool_;

Coordinator::Coordinator() {} // call이 안된다.

Coordinator::~Coordinator() {}

bool Coordinator::Coordination(long long step_id, int node_id, std::string node_name, Device* device) {
  if (!initialized) {
    Initialization(device);
  }

  /*
    Align first step(start aligning step) only!
    For the rest of them, just run.
  */
  if (coordinator_policy == "AlignFisrtStepOnly") {
    return Policy_0(step_id, node_id, node_name, device);
  }

  /*
    Profiled peak memory node name + 90 percentile memory consumption in terms of peak
  */
  else if (coordinator_policy == "PeakNodeName") {
    return Policy_1(step_id, node_id, node_name, device);
  }

  /*
    95 percentile peak memory based backward starting point detection
  */
  else if (coordinator_policy == "95PercentilePeakMemeory") {
    return Policy_2(step_id, node_id, node_name, device);
  }
  
  /*
    Last BFC Allocation (or Last BFC Deallocation) based back-to-back scheduling.

    When Job A finishes an iteration, start Job B's iteration.
    Control Job B only. Job A will not be blocked by Coordinator.

        AA        BB      AA        BB
       A  A      B  B    A  A      B  B
      A    A -> B    B  A    A -> B    B
  */
  else if (coordinator_policy == "LastDeallocation") {
    return Policy_3(step_id, node_id, node_name, device);
  }

  /*
    Execution order based backward detection
    Profiling the execution id that triggers peak memory
  */
  else if (coordinator_policy == "PeakExecutionId") {
    return Policy_4(step_id, node_id, node_name, device);
  }

  /*
    Coordinating 4 jobs
  */
  else if (coordinator_policy == "NaiveTwoPlusTwo") {
    return Policy_5(step_id, node_id, node_name, device);
  }

  /*
    Iteration back-to-back scheduling
    Last execution id 기반
  */
  else if (coordinator_policy == "LastExecutionId") {
    return Policy_6(step_id, node_id, node_name, device);
  }

  /*
    Model Based Scheduler (글로벌 풀에 있는거 옮겨보는 중)
  */
  else if (coordinator_policy == "ModelBased_Coordinator") {
    return Policy_8(step_id, node_id, node_name, device);
  }
}

void Coordinator::print_current_state(std::string allocator_name,
                                      std::string cur_state, long long step_id,
                                      int node_id, std::string node_name) {
  std::cout << "(" << allocator_name << ") " << "(step_id: " << step_id << ") "
            << cur_state << ", id," << node_id << " name," << node_name
            << ", thread_id," << std::this_thread::get_id() << std::endl;
  fflush(stdout);
}

} // namespace tensorflow
