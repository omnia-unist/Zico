#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GANGMUK_THREADPOOL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GANGMUK_THREADPOOL_H_

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace tensorflow {

class GM_ThreadPool {
 public:
  GM_ThreadPool();
  GM_ThreadPool(size_t num_threads);
  ~GM_ThreadPool();

  // job 을 추가한다.
  void EnqueueJob(std::function<void()> job);

 private:
  // 총 Worker 쓰레드의 개수.
  size_t num_threads_;
  // Worker 쓰레드를 보관하는 벡터.
  std::vector<std::thread> worker_threads_;
  // 할일들을 보관하는 job 큐.
  std::queue<std::function<void()>> jobs_;
  // 위의 job 큐를 위한 cv 와 m.
  std::condition_variable cv_job_q_;
  std::mutex m_job_q_;

  // 모든 쓰레드 종료
  bool stop_all;

  // Worker 쓰레드
  void WorkerThread();
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GANGMUK_THREADPOOL_H_