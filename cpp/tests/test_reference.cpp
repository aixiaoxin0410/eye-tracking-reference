// Keep behavioral assertions active when CMake uses Release/NDEBUG.
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "calibration_session.hpp"
#include "stereo_worker.hpp"
#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

struct CountingDetector {
    using Input = int;
    using Output = std::pair<int, std::thread::id>;
    int counter = 0;
    Output operator()(const Input& value) {
        if (value < 0) throw std::runtime_error("synthetic detector failure");
        return {value + ++counter, std::this_thread::get_id()};
    }
};

int main() {
    using namespace eye_reference;
    StereoWorker<CountingDetector> worker(CountingDetector{}, CountingDetector{});
    const auto first = worker.process(10, 100);
    assert(first[0].first == 11 && first[1].first == 101);
    assert(first[0].second != first[1].second);
    for (int i = 2; i <= 40; ++i) {
        const auto result = worker.process(10, 100);
        assert(result[0].first == 10 + i && result[1].first == 100 + i);
        assert(result[0].second == first[0].second);
        assert(result[1].second == first[1].second);
    }
    bool failed = false;
    try { worker.process(-1, 100); } catch (const std::runtime_error&) { failed = true; }
    assert(failed);
    const auto recovered = worker.process(10, 100);
    assert(recovered[0].first == 51 && recovered[1].first == 142);

    CalibrationSession calibration;
    StereoFeatures frame{{true, false, 1, 2, 1}, {true, false, 3, 4, 1}};
    assert(!calibration.add(frame));
    std::int64_t timestamp = 1;
    for (int point = 0; point < 9; ++point) {
        calibration.begin_point(point, {0.15 + (point % 3)*0.35, 0.15 + (point/3)*0.35});
        frame.left.timestamp_us = timestamp;
        frame.right.timestamp_us = timestamp + 1;
        assert(!calibration.add(frame)); // Unsynchronized samples cannot train a model.
        for (int i = 0; i < 3; ++i) {
            frame.left.timestamp_us = frame.right.timestamp_us = timestamp++;
            assert(calibration.add(frame));
            assert(!calibration.add(frame)); // Repeated timestamp.
        }
        frame.left.blink = true;
        assert(!calibration.add(frame));
        frame.left.blink = false;
        calibration.end_point();
    }
    assert(calibration.ready(3));
    assert(!calibration.ready(4));
    assert(calibration.samples()[8][0].target.x == 0.85);
    assert(calibration.samples()[8][0].point_index == 8);
    calibration.reset();
    assert(!calibration.ready(1));
    std::cout << "PASS: isolated persistent workers, exception recovery, labeled calibration, timestamps\n";
}
