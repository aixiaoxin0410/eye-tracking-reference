#pragma once

#include <array>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace eye_reference {
// Detector provides Input, Output and Output operator()(const Input&).
// Every eye owns a distinct detector and a persistent worker. Detector state may
// be mutable; do not share mutable state through external pointers or globals.
// Input copies must retain immutable storage until process() returns.
// Destruction must not overlap a process() call, as with normal owning objects.
template<class Detector>
class StereoWorker {
public:
    using Input = typename Detector::Input;
    using Output = typename Detector::Output;
    using Pair = std::array<Output, 2>;

    StereoWorker(Detector left, Detector right)
        : detectors_{std::move(left), std::move(right)} {
        // Start threads in the body: every mutex/condition variable is initialized.
        try {
            workers_[0] = std::thread([this] { loop(0); });
            workers_[1] = std::thread([this] { loop(1); });
        } catch (...) {
            stop();
            throw;
        }
    }
    ~StereoWorker() { stop(); }
    StereoWorker(const StereoWorker&) = delete;
    StereoWorker& operator=(const StereoWorker&) = delete;

    Pair process(Input left, Input right) {
        std::lock_guard<std::mutex> caller_lock(caller_mutex_);
        std::unique_lock<std::mutex> state_lock(state_mutex_);
        inputs_[0] = std::move(left);
        inputs_[1] = std::move(right);
        results_[0].reset(); results_[1].reset();
        errors_ = {};
        completed_ = 0;
        ++generation_;
        job_cv_.notify_all();
        done_cv_.wait(state_lock, [this] { return completed_ == 2; });
        for (const auto& error : errors_) if (error) std::rethrow_exception(error);
        return {std::move(*results_[0]), std::move(*results_[1])};
    }
private:
    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            stopping_ = true;
        }
        job_cv_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }
    void loop(std::size_t eye) {
        std::size_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(state_mutex_);
            job_cv_.wait(lock, [this, &seen] { return stopping_ || generation_ != seen; });
            if (stopping_) return;
            seen = generation_;
            // Each thread accesses a separate array element. process() waits for
            // both workers before touching inputs/results again.
            lock.unlock();
            try { results_[eye] = detectors_[eye](*inputs_[eye]); }
            catch (...) { errors_[eye] = std::current_exception(); }
            lock.lock();
            ++completed_;
            if (completed_ == 2) done_cv_.notify_one();
        }
    }
    std::array<Detector, 2> detectors_;
    std::mutex caller_mutex_, state_mutex_;
    std::condition_variable job_cv_, done_cv_;
    bool stopping_ = false;
    std::size_t generation_ = 0;
    int completed_ = 0;
    std::array<std::optional<Input>, 2> inputs_;
    std::array<std::optional<Output>, 2> results_;
    std::array<std::exception_ptr, 2> errors_;
    std::array<std::thread, 2> workers_;
};
}  // namespace eye_reference
