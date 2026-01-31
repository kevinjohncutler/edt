/* 
Copyright (c) 2012 Jakob Progsch, Václav Zeman

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

   1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.

   2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.

   3. This notice may not be removed or altered from any source
   distribution.

Notice of Alteration
William Silversmith
May 2019, December 2023

- The license file was moved from a seperate file to the top of this one.
- Created public "join" member function from destructor code.
- Created public "start" member function from constructor code.
- Used std::invoke_result_t to update to modern C++
*/

#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

class ThreadPool {
public:
    ThreadPool(size_t);
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>;
    void start(size_t);
    void join();
    void wait();  // Wait for all tasks to complete without destroying threads
    ~ThreadPool();
private:
    // need to keep track of threads so we can join them
    std::vector< std::thread > workers;
    // the task queue
    std::queue< std::function<void()> > tasks;

    // synchronization
    std::mutex queue_mutex;
    std::condition_variable condition;
    std::condition_variable done_condition;  // Signals when queue is empty and all workers idle
    bool stop;
    size_t active_tasks;  // Number of tasks currently being executed
};
 
// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads)
    :   stop(false), active_tasks(0)
{
    start(threads);
}

void ThreadPool::start(size_t threads) {
    stop = false;
    active_tasks = 0;
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(
            [this]
            {
                for(;;)
                {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                            [this]{ return this->stop || !this->tasks.empty(); });
                        if(this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                        this->active_tasks++;
                    }

                    task();

                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->active_tasks--;
                        if (this->tasks.empty() && this->active_tasks == 0) {
                            this->done_condition.notify_all();
                        }
                    }
                }
            }
        );
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<std::invoke_result_t<F, Args...>>
{
    using return_type = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        // don't allow enqueueing after stopping the pool
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

inline void ThreadPool::join () {
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    workers.clear();
    // clear any remaining tasks to avoid destructor-time aborts
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        while(!tasks.empty()) tasks.pop();
    }
}

// Wait for all current tasks to complete without destroying threads
inline void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(queue_mutex);
    done_condition.wait(lock, [this]{
        return tasks.empty() && active_tasks == 0;
    });
}

// the destructor joins all threads
inline ThreadPool::~ThreadPool() {
    join();
}



#endif
