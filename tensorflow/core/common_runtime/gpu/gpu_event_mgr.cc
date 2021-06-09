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

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"

#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

namespace {
// The EventMgr has 1 thread for the polling loop and one to execute
// event callback functions. Issues for reconsideration:
//  - Is this the right number of threads?
//  - Should EventMgrs be shared between GPUDevices on a multi-GPU machine?
static const int kNumThreads = 2;
}  // namespace

namespace gpu_event_mgr {
class ThreadLabel {
 public:
  static const char* GetValue() { return value_; }

  // v must be a static const because value_ will capture and use its value
  // until reset or thread terminates.
  static void SetValue(const char* v) { value_ = v; }

 private:
  static thread_local const char* value_;
};
thread_local const char* ThreadLabel::value_ = "";

void WarnIfInCallback(std::function<void()> f) {
  const char* label = ThreadLabel::GetValue();
  if (label && !strcmp(label, "gpu_event_mgr")) {
    if (f) {
      f();
    } else {
      LOG(WARNING) << "Executing inside EventMgr callback thread: "
                   << CurrentStackTrace();
    }
  }
}

void InitThreadpoolLabels(thread::ThreadPool* threadpool) {
  static const char* label = "gpu_event_mgr";
  mutex mu;
  int init_count = 0;
  condition_variable all_initialized;
  int exit_count = 0;
  condition_variable ready_to_exit;
  const int num_threads = threadpool->NumThreads();
  for (int i = 0; i < num_threads; ++i) {
    threadpool->Schedule([num_threads, &mu, &init_count, &all_initialized,
                          &exit_count, &ready_to_exit]() {
      gpu_event_mgr::ThreadLabel::SetValue(label);
      mutex_lock l(mu);
      ++init_count;
      if (init_count == num_threads) {
        all_initialized.notify_all();
      }
      while (init_count < num_threads) {
        all_initialized.wait(l);
      }
      if (++exit_count == num_threads) {
        ready_to_exit.notify_all();
      }
    });
  }
  {
    mutex_lock l(mu);
    while (exit_count < num_threads) {
      ready_to_exit.wait(l);
    }
  }
}
}  // namespace gpu_event_mgr

// Gangmuk
mutex EventMgr::mu_gangmuk;

// Original Constructor
EventMgr::EventMgr(se::StreamExecutor* se, const GPUOptions& gpu_options)
    : exec_(se),
      deferred_bytes_threshold_(gpu_options.deferred_deletion_bytes()
                                    ? gpu_options.deferred_deletion_bytes()
                                    : 8 * 1048576),
      polling_active_delay_usecs_(gpu_options.polling_active_delay_usecs()
                                      ?
                                      gpu_options.polling_active_delay_usecs()
                                      : 10),
      accumulated_stream_(nullptr),
      accumulated_tensors_(new TensorReferenceVector),
      accumulated_tensor_bytes_(0),
      threadpool_(Env::Default(), "GPU_Event_Manager", kNumThreads) {
      // EventRecord_threadpool_(Env::Default(), "GPU_Event_Manager_EventRecord", 1) {
  gpu_event_mgr::InitThreadpoolLabels(&threadpool_);
  // gpu_event_mgr::InitThreadpoolLabels(&EventRecord_threadpool_);
  StartPollingLoop();
}

// Gangmuk
EventMgr::EventMgr(se::StreamExecutor* se, long long deferred_bytes_threshold,
                   int polling_active_delay_usecs, bool low_latency_hint,
                   int kNumThreads_gangmuk)
    : exec_(se),
      deferred_bytes_threshold_(deferred_bytes_threshold),
      polling_active_delay_usecs_(polling_active_delay_usecs),
      accumulated_stream_(nullptr),
      accumulated_tensors_(new TensorReferenceVector),
      accumulated_tensor_bytes_(0),
      threadpool_(Env::Default(), "GPU_Event_Manager", kNumThreads) {
      // EventRecord_threadpool_(Env::Default(), "GPU_Event_Manager_EventRecord", 1) {
  if (deferred_bytes_threshold_ == -1) {
    deferred_bytes_threshold_ = 8 << 20; // 8MB, default value
    printf("deferred_bytes_threshold_ = %lld, set to default value\n",
           deferred_bytes_threshold_);
  }

  if (polling_active_delay_usecs_ == -1) {
    polling_active_delay_usecs_ = 10; // 10us, default value
    printf("polling_active_delay_usecs_ = %lld, set to default value\n",
           polling_active_delay_usecs_);
  }

  printf("---------------- gpu_event_mgr::InitThreadpoolLabels(&threadpool_)\n");
  gpu_event_mgr::InitThreadpoolLabels(&threadpool_);
  // printf("---------------- gpu_event_mgr::InitThreadpoolLabels(&EventRecord_threadpool_)\n");
  // gpu_event_mgr::InitThreadpoolLabels(&EventRecord_threadpool_);
  StartPollingLoop();
}

EventMgr::~EventMgr() {
  StopPollingLoop();
  printf("EventMgr Destructor\n");
  // Events are owned by this object.
  for (auto& e : free_events_) {
    delete e;
  }
  for (auto& t : *(accumulated_tensors_)) {
    t.Unref();
  }
  delete accumulated_tensors_;
  while (!used_events_.empty()) {
    InUse* ue = &used_events_[0];
    delete ue->event;
    if (ue->mem != nullptr) {
      for (auto& t : *(ue->mem)) {
        t.Unref();
      }
      delete ue->mem;
    }
    if (ue->bufrec.buf) {
      if (LogMemory::IsEnabled()) {
        LogMemory::RecordRawDeallocation(ue->bufrec.operation,
                                         ue->bufrec.step_id, ue->bufrec.buf,
                                         ue->bufrec.alloc, false);
      }
      ue->bufrec.alloc->DeallocateRaw(ue->bufrec.buf);
    }
    if (ue->func != nullptr) threadpool_.Schedule(ue->func);
    used_events_.pop_front();
  }
  printf("End of EventMgr Destructor\n");
}

void EventMgr::StartPollingLoop() {
  CHECK(polling_stopped_ == nullptr);
  {
    mutex_lock l(mu_);
    stop_polling_ = false;
  }
  polling_stopped_.reset(new Notification);
  printf("---------------- threadpool_.Schedule([this]() { PollLoop(); })\n");
  threadpool_.Schedule([this]() { PollLoop(); });
}

void EventMgr::StopPollingLoop() {
  if (polling_stopped_) {
    {
      mutex_lock l(mu_);
      stop_polling_ = true;
      events_pending_.notify_all();
    }
    polling_stopped_->WaitForNotification();
    polling_stopped_.reset(nullptr);
  }
}

void EventMgr::ThenDeleteTensors(se::Stream* stream,
                                 const TensorReferenceVector& tensors) {
  mutex_lock l(mu_);
  ++num_pending_events_;
  // TODO(jeff): We currently keep one accumulated_tensors_ object.
  // If we start to use multiple streams heavily, we might want to keep
  // separate vectors/byte counters per stream
  if (!accumulated_tensors_->empty() && stream != accumulated_stream_) {
    printf("!accumulated_tensors_->empty()\n"); // 여기는 출력 안됨.
    FlushAccumulatedTensors();
  }
  accumulated_stream_ = stream;
  ///////////////////////////////////////////////////////////////////////
  /**
   * tensor 쌓는 부분
   **/
  for (const auto& t : tensors) {
    // accumulated_tensors_ takes over ownership of the reference to "t"
    accumulated_tensors_->push_back(t);
    accumulated_tensor_bytes_ += t.TotalBytes();
  }
  ///////////////////////////////////////////////////////////////////////
  
  // Gangmuk
  if (accumulated_tensor_bytes_ >= deferred_bytes_threshold_) {
    // 기존 방법, 아무것도 안하기
    FlushAccumulatedTensors();

    // 시도1: naive threading with join, 작동하지만, 뭐 당연한거고, 느리다.
    // {
    //   std::thread t1 = std::thread([this]() { FlushAccumulatedTensors(); });
    //   t1.join();
    // }

    // 시도2: naive threading with detach, [Seg Fault]
    // {
    //   std::thread t1 = std::thread([this]() { FlushAccumulatedTensors(); });
    //   t1.detach(); // seg fault
    // }

    // 시도3: TF ThreadPool, [Seg Fault]
    // {
    //   mutex_lock l(mu_gangmuk); // lock이 제대로 안되는 거 처럼 보임.
    //   mutex_lock l(mu_); // deadlock 발생
    //   printf("---------------- EventRecord_threadpool_.Schedule([this]() {
    //   FlushAccumulatedTensors(); })\n"); threadpool_.Schedule([this]() {
    //   FlushAccumulatedTensors(); }); // seg fault
    //   EventRecord_threadpool_.Schedule([this]() { FlushAccumulatedTensors();
    //   }); // seg fault
    // }

    // // 시도4: Custom ThreadPool, [Seg Fault]
    // gm_threadpool.EnqueueJob([this]() { FlushAccumulatedTensors(); });
  }
}

void EventMgr::FlushAccumulatedTensors() {
  // mutex_lock l(mu_gangmuk);
  // printf("---------------- FlushAccumulatedTensors\n");
  DCHECK(!accumulated_tensors_->empty());
  DCHECK(accumulated_stream_ != nullptr);
  QueueTensors(accumulated_stream_, accumulated_tensors_);
  accumulated_tensors_ = new TensorReferenceVector;
  accumulated_tensor_bytes_ = 0;
  accumulated_stream_ = nullptr;
}

// A polling loop to detect completion of GPU events.
//
// While one or more events is outstanding, poll for completed events.  When no
// events are outstanding, we sleep until one is enqueued.
void EventMgr::PollLoop() {
  ToFreeVector to_free;
  while (true) {
    bool events_still_pending;
    {
      mutex_lock l(mu_);
      if (stop_polling_) {
        break;
      }
      if (used_events_.empty()) {
        events_pending_.wait(l);
      }
      PollEvents(true, &to_free);
      // printf("PollEvents,%d,%ld\n", pthread_self(), gettid());
      events_still_pending = !used_events_.empty();
      // "이벤트들을 폴을 한 뒤에도 아직 컴플리션이 안되고 펜딩되어있는 이벤트가 남아있으면..."
      // 그러면, used_events_ 가 펜딩되어있는 이벤트들을 담고있는 FIFO Queue인 듯 => 맞음
    }
    FreeMemory(to_free);
    to_free.clear();

    if (events_still_pending) {
      // 그런데, 그러면 이해가 잘 안되네. 왜 컴플리션이 안되고 펜딩된
      // 이벤트가 여전히 있으면 슬립을 하지?
      // 반대여야할 거 같은데. 펜딩된 이벤트가 없으면 큐에 이벤트가 들어올
      // 때까지 기다리는 로직이어야하는거 아닌가?
      // 아! 이건 폴을 했음에도 여전히 펜딩이라는 것은 바로 폴을 시도하지 말고,
      // 어떤 이벤트가 컴플리션이 될 때까지 좀 기다렸다가 다시 폴을 시도하라는 뜻이구나!
      // polling_active_delay_usecs_: 10us
      Env::Default()->SleepForMicroseconds(polling_active_delay_usecs_);
    }
  }
  polling_stopped_->Notify();
}

void EventMgr::QueueInUse(se::Stream* stream, InUse iu) {
  VLOG(2) << "QueueInUse  free_events_ " << free_events_.size()
          << " used_events_ " << used_events_.size();
  // Events are created on demand, and repeatedly reused. There is no
  // limit placed here on the number of allocated Events.
  int i = 0;
  // printf("QueueInUse %d\n",i++);
  if (free_events_.empty()) {
    // printf("QueueInUse free_events_.empty()\n");
    free_events_.push_back(new se::Event(exec_));
    free_events_.back()->Init();
  }
  // printf("QueueInUse %d\n", i++);
  se::Event* e = free_events_.back();
  // printf("QueueInUse %d\n", i++);
  free_events_.pop_back();
  // printf("QueueInUse %d\n", i++);
  // if (stream == nullptr) {
    // printf("-------- QueueInUse, stream == nullptr\n");
  //  exit(-1);
  // }
  stream->ThenRecordEvent(e); // seg fault
  // printf("QueueInUse %d\n", i++);
  iu.event = e;
  // printf("QueueInUse %d\n", i++);
  bool was_empty = used_events_.empty();
  // printf("QueueInUse %d\n", i++);
  used_events_.push_back(iu);
  // printf("QueueInUse %d\n", i++);
  // Maybe wake up the polling thread
  if (was_empty) events_pending_.notify_all();
  // printf("QueueInUse %d\n", i++);
}

// This function must be called periodically to check whether pending
// events have recorded, and then retire them.  Initial observations
// suggest that typical behavior in a TensorFlow program is to have
// 0-3 events pending most of the time, but there are occasionally
// spikes of up to several hundred outstanding.
//
// NOTE: If all events are on the same stream, no later event will
// complete before an earlier event, except possibly if the earlier
// event transitions to an error state, so there's no advantage in
// looking past the first kPending event.  However, if we're using
// multiple streams there may be some gain in looking deeper.
// As a compromise, PollEvent() calls that are triggered by the queueing
// of a single event never look past the first kPending event.  Calls
// coming from the dedicated polling thread always sweep the full queue.
//
// Note that allowing the queue to grow very long could cause overall
// GPU memory use to spike needlessly.  An alternative strategy would
// be to throttle new Op execution until the pending event queue
// clears.
void EventMgr::PollEvents(bool is_dedicated_poller,
                          gtl::InlinedVector<InUse, 4>* to_free) {
  VLOG(2) << "PollEvents  free_events_ " << free_events_.size()
          << " used_events_ " << used_events_.size();
  // Sweep the remaining events in order.  If this is the dedicated
  // polling thread, check the entire set.  Otherwise, just sweep up to
  // the first non-complete record that is still pending.
  for (auto& iu : used_events_) {
    if (iu.event == nullptr) continue;
    se::Event::Status s = iu.event->PollForStatus();
    switch (s) {
      case se::Event::Status::kUnknown:
      case se::Event::Status::kError:
        // We don't expect to see these.  Someday maybe propagate
        // a Status error, but for now fail hard.
        LOG(FATAL) << "Unexpected Event status: " << static_cast<int>(s);
        break;
      case se::Event::Status::kPending:
        // is_dedicated_poller is always true. Hence, no return here.
        // 따라서, 여기서 리턴하지 않고 계속 루프를 돌면서, 
        // 큐에 있는 이벤트 전체를 다 볼 것이다.
        if (!is_dedicated_poller) return;  // quit processing queue
        break;
      case se::Event::Status::kComplete:
        // Make a copy of the InUse record so we can free it after releasing
        // the lock
        to_free->push_back(iu);
        free_events_.push_back(iu.event);
        // Mark this InUse record as completed.
        iu.event = nullptr;
    }
  }
  // Then clear any completed InUse records from the front of the queue.
  while (!used_events_.empty()) {
    InUse& iu = used_events_.front();
    if (iu.event == nullptr) {
      used_events_.pop_front();
    } else {
      break;
    }
  }

  // ", ThreadId," << std::this_thread::get_id() << std::endl;
}

}  // namespace tensorflow
