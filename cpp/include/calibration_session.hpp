#pragma once

#include "eye_types.hpp"
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace eye_reference {
// Event-driven collection only. Model fitting belongs to the caller.
// Labels are stored per sample; point identity is never inferred from frame count.
class CalibrationSession {
public:
    static constexpr std::size_t point_count = 9;
    void reset() {
        for (auto& batch : samples_) batch.clear();
        active_ = -1;
        last_timestamp_ = -1;
    }
    void begin_point(int index, ScreenPoint target) {
        if (active_ >= 0) throw std::logic_error("End current point before starting another");
        if (index < 0 || index >= static_cast<int>(point_count))
            throw std::out_of_range("Expected point index in [0,8]");
        if (!std::isfinite(target.x) || !std::isfinite(target.y) ||
            target.x < 0 || target.x > 1 || target.y < 0 || target.y > 1)
            throw std::invalid_argument("Target must be a normalized screen point");
        active_ = index;
        target_ = target;
        samples_[index].clear();
    }
    bool add(const StereoFeatures& frame) {
        if (active_ < 0 || !frame.left.usable() || !frame.right.usable() ||
            frame.left.timestamp_us != frame.right.timestamp_us ||
            frame.left.timestamp_us <= last_timestamp_) return false;
        samples_[active_].push_back({active_, target_, frame});
        last_timestamp_ = frame.left.timestamp_us;
        return true;
    }
    void end_point() { active_ = -1; }
    bool ready(std::size_t minimum_per_point = 20) const {
        if (active_ >= 0 || minimum_per_point == 0) return false;
        for (const auto& batch : samples_)
            if (batch.size() < minimum_per_point) return false;
        return true;
    }
    const std::array<std::vector<CalibrationSample>, point_count>& samples() const {
        return samples_;
    }
private:
    std::array<std::vector<CalibrationSample>, point_count> samples_;
    int active_ = -1;
    std::int64_t last_timestamp_ = -1;
    ScreenPoint target_;
};
}  // namespace eye_reference
