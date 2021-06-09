#include "tensorflow/core/common_runtime/gpu/gangmuk_threadpool.h"
// #include "gangmuk_threadpool.h"

namespace tensorflow {

GM_ThreadPool::GM_ThreadPool()
    : num_threads_(1), stop_all(false) {
  worker_threads_.reserve(1);
  for (size_t i = 0; i < num_threads_; ++i) {
    printf("---- GM_ThreadPool %d\n", i);
    worker_threads_.emplace_back([this]() { this->WorkerThread(); });
  }
}

GM_ThreadPool::GM_ThreadPool(size_t num_threads)
    : num_threads_(num_threads), stop_all(false) {
  worker_threads_.reserve(num_threads_);
  for (size_t i = 0; i < num_threads_; ++i) {
    worker_threads_.emplace_back([this]() { this->WorkerThread(); });
  }
}

void GM_ThreadPool::WorkerThread() {
  while (true) {
    std::unique_lock<std::mutex> lock(m_job_q_);
    cv_job_q_.wait(lock, [this]() { return !this->jobs_.empty() || stop_all; });
    if (stop_all && this->jobs_.empty()) {
      return;
    }

    // 맨 앞의 job 을 뺀다.
    std::function<void()> job = std::move(jobs_.front());
    jobs_.pop();
    lock.unlock();

    // 해당 job 을 수행한다 :)
    job();
  }
}

GM_ThreadPool::~GM_ThreadPool() {
  stop_all = true;
  cv_job_q_.notify_all();

  for (auto& t : worker_threads_) {
    t.join();
  }
}

void GM_ThreadPool::EnqueueJob(std::function<void()> job) {
  if (stop_all) {
    // throw std::runtime_error("GM_ThreadPool 사용 중지됨");
    printf("GM_ThreadPool 사용 중지됨\n");
    exit(-1);
  }
  {
    std::lock_guard<std::mutex> lock(m_job_q_);
    jobs_.push(std::move(job));
  }
  cv_job_q_.notify_one();
}

// void work(int t, int id) {
//   printf("%d start \n", id);
//   std::this_thread::sleep_for(std::chrono::seconds(t));
//   printf("%d end after %ds\n", id, t);
// }

// int main() {
//   GM_ThreadPool pool(1);

//   for (int i = 0; i < 10; i++) {
//     pool.EnqueueJob([i]() { work(i % 3 + 1, i); });
//   }
//   return 0;
// }

}  // namespace tensorflow
