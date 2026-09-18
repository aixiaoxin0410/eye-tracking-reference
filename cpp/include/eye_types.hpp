#pragma once

// Public, dependency-free reference types. No image pixels cross this interface.
#include <cmath>
#include <cstdint>

namespace eye_reference {
struct EyeFeature {
    bool valid = false;
    bool blink = false;
    double dx = 0;
    double dy = 0;
    std::int64_t timestamp_us = 0;
    bool usable() const {
        return valid && !blink && std::isfinite(dx) && std::isfinite(dy);
    }
};
struct StereoFeatures { EyeFeature left, right; };
struct ScreenPoint { double x = 0.5, y = 0.5; };
struct CalibrationSample {
    int point_index = -1;
    ScreenPoint target;
    StereoFeatures features;
};
}  // namespace eye_reference
