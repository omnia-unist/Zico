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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_

#include <deque>
#include <thread>
#include <vector>

#include "tensorflow/core/framework/log_memory.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_reference.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/common_runtime/gpu/gangmuk_threadpool.h"

#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)
#endif

namespace stream_executor {
class Event;
class Stream;
class StreamExecutor;
}  // namespace stream_executor

namespace tensorflow {

class GPUOptions;

// The callback provided to EventMgr::ThenExecute must not block or take a long
// time.  If it does, performance may be impacted and GPU memory may be
// exhausted.  This macro is for checking that an EventMgr thread is not
// accidentally entering blocking parts of the code, e.g. the RPC subsystem.
//
// Intended use is something like
//
//   void RespondToAnRPC(Params* params) {
//      WARN_IF_IN_EVENT_MGR_THREAD;
//      if (params->status.ok()) { ...
//
namespace gpu_event_mgr {
// Logs a stack trace if current execution thread belongs to this EventMgr
// object.  If f is not nullptr, executes instead of  logging the stack trace.
// trace.
void WarnIfInCallback(std::function<void()> f);
}  // namespace gpu_event_mgr
#define WARN_IF_IN_EVENT_MGR_THREAD gpu_event_mgr::WarnIfInCallback(nullptr)

// An object to keep track of pending Events in the StreamExecutor streams
// and associated Tensors that cannot safely be deleted until the associated
// Events are recorded.
class EventMgr {
 public:
  EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options);
  // Gangmuk
  EventMgr(se::StreamExecutor* se, long long deferred_deletion_bytes,
           int polling_active_delay_usecs, bool low_latency_hint,
           int kNumThreads_gangmuk);

  ~EventMgr();

  // Gangmuk
  // GM_Parser parser;
  bool print_EventMgr = false;
  GM_ThreadPool gm_threadpool;
  unsigned long long event_id = 0;

  // Releases the references on the elements of "tensors" as soon as
  // all events currently enqueued on "stream" have completed.
  void ThenDeleteTensors(se::Stream* stream,
                         const TensorReferenceVector& tensors);

  struct BufRec {
    Allocator* alloc;
    void* buf;
    // operation and step_id are only populated when
    // LogMemory::IsEnabled() is true.
    string operation;
    int64 step_id;
  };

  // Gangmuk: 이거는 콜이 안된다.
  // Takes ownership of *bufrec.buf and calls bufrec.alloc->DeallocateRaw()
  // on it as soon as all events currently enqueued on *stream have completed.
  inline void ThenDeleteBuffer(se::Stream* stream, BufRec bufrec) {
    ToFreeVector to_free;
    {
      mutex_lock l(mu_);
      QueueBuffer(stream, bufrec);
      PollEvents(false, &to_free);
    }
    FreeMemory(to_free);
  }

  // Execute func when all pending stream actions have completed.
  // func must be brief and non-blocking since it executes in the one
  // thread used for all such callbacks and also buffer deletions.
  inline void ThenExecute(se::Stream* stream, std::function<void()> func) {
    ToFreeVector to_free;
    {
      mutex_lock l(mu_);
      QueueFunc(stream, std::move(func));
      PollEvents(false, &to_free);
    }
    FreeMemory(to_free);
  }

 private:
  friend class TEST_EventMgrHelper;
  se::StreamExecutor* const exec_;
  // const int64 deferred_bytes_threshold_;
  // const int32 polling_active_delay_usecs_;
  int64 deferred_bytes_threshold_;
  int32 polling_active_delay_usecs_;
  mutex mu_;
  static mutex mu_gangmuk;
  condition_variable events_pending_ GUARDED_BY(mu_);

  void FlushAccumulatedTensors() EXCLUSIVE_LOCKS_REQUIRED(mu_);

  struct InUse {
    se::Event* event;
    TensorReferenceVector* mem;
    BufRec bufrec;
    std::function<void()> func;
  };

  typedef gtl::InlinedVector<InUse, 4> ToFreeVector;

  void FreeMemory(const ToFreeVector& to_free) {
    for (const auto& iu : to_free) {
      if (iu.mem != nullptr) {
        for (auto& t : *(iu.mem)) {
          t.Unref(); // 이 Unref()로 인해서 무조건(매번) Deallocate가 trigger 되는 건 아니다.
        }
        delete iu.mem;
      }
      if (iu.bufrec.buf) {
        // Gangmuk: 여기는 도달 안하는 것으로 보임.
        printf("########## iu.bufrec.buf is Not Null\n");
        exit(-1);
        if (LogMemory::IsEnabled()) {
          LogMemory::RecordRawDeallocation(iu.bufrec.operation,
                                           iu.bufrec.step_id, iu.bufrec.buf,
                                           iu.bufrec.alloc, false);
        }
        iu.bufrec.alloc->DeallocateRaw(iu.bufrec.buf); // Original Code
      }
      // The function must be called in another thread.
      if (iu.func != nullptr) {
        threadpool_.Schedule(iu.func);
      }
    }
  }

  // Stream-enqueue an unused Event and save with it a collection of
  // Tensors and/or a BufRec to be deleted only after the Event
  // records.
  void QueueInUse(se::Stream* stream, InUse in_use)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void QueueTensors(se::Stream* stream, TensorReferenceVector* tensors)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    int i = 0;
    // printf("QueueTensors %d\n", i++);
    QueueInUse(stream, {nullptr, tensors, BufRec(), nullptr});
    // printf("QueueTensors %d\n", i++);
  }

  void QueueBuffer(se::Stream* stream, BufRec bufrec)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, nullptr, bufrec, nullptr});
  }

  void QueueFunc(se::Stream* stream, std::function<void()> func)
      EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    QueueInUse(stream, {nullptr, nullptr, BufRec(), std::move(func)});
  }

  // This function should be called at roughly the same tempo as
  // QueueTensors() to check whether pending events have recorded,
  // and then retire them.  It appends InUse elements that need cleanup
  // to "*to_free".  The caller should call FreeMemory(to_free)
  // when this returns.
  void PollEvents(bool is_dedicated_poller, ToFreeVector* to_free)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // An internal polling loop that runs at a low frequency to clear
  // straggler Events.
  void PollLoop();

  // Setup/Teardown functions for the polling loop.
  void StartPollingLoop();
  void StopPollingLoop();

  // A stack of unused events
  std::vector<se::Event*> free_events_ GUARDED_BY(mu_);

  // Buffered list of tensors waiting to have an event queued for deletion
  se::Stream* accumulated_stream_ GUARDED_BY(mu_);
  TensorReferenceVector* accumulated_tensors_ GUARDED_BY(mu_);
  // Sum of the TotalBytes() of the tensors in "accumulated_tensors_"
  int64 accumulated_tensor_bytes_ GUARDED_BY(mu_);

  bool stop_polling_ GUARDED_BY(mu_);
  std::unique_ptr<Notification> polling_stopped_;

  // The main PollLoop for the event manager runs in this threadpool.
  thread::ThreadPool threadpool_;

  // Gangmuk
  // Another threadpool for EventRecording
  // thread::ThreadPool EventRecord_threadpool_;

  public:
    // Gangmuk: Moved to public
    // A FIFO queue of InUse events and associated tensors.
    std::deque<InUse> used_events_ GUARDED_BY(mu_);
    int num_pending_events_ = 0;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_EVENT_MGR_H_
