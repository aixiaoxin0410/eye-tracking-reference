# MediaPipe / device integration sketch

本目录只说明接入接口，`eye_tracking_graph.pbtxt` 不属于可运行示例或 CI 构建目标。图中 Calculator 名称是接入契约，占位实现未随仓库发布。

## 需要补齐的模块

| 模块 | 预期职责 |
| --- | --- |
| Camera / FaceLandmark / EyeRoi | 生成同一时间戳的双眼 ROI，明确灰度格式、步长与坐标系 |
| PupilGlintCalculator | 每实例独立检测状态，输出 valid / blink / dx / dy / timestamp |
| CalibrationModel | 对四维同分布原始特征进行预测，并持久化版本、坐标约定和校验和 |
| GazeSolverCalculator | 同步双眼；无效帧标记、模型预测和最终滤波；保留输入时间戳 |
| Device output | 发送归一化视线坐标与有效性、过期状态 |

OpenCV 图像包装可使用只读 ROI 视图，但调用方必须保证像素生命周期；不能仅靠 `const cv::Mat&` 认为底层内容不会被修改。图中两个瞳孔节点无依赖，只有在 executor 有足够线程时才能并行调度，图结构本身不保证提速。

## 标定事件契约

```text
reset → display target → acknowledge visible → wait for settling
      → begin_point(index, x, y) → collect valid synchronized pairs
      → end_point → next target → verify sample coverage → fit → save
```

每帧保留点编号与显式目标，允许各点样本数不同。`cpp/include/calibration_session.hpp` 只实现收集；稳定等待、最少有效帧数、离群值策略、训练与持久化由集成方实现。Python 演示直接生成稳定注视样本，没有模拟显示延迟或设备协议。

建议设备间使用经过同步的单调时间基准；显示端时间与摄像头时间若来源不同，不能直接相减。单眼降级需要独立训练单眼模型，不能将缺失眼特征填零后直接送入双眼模型。

## 可验证的优化方向

- 固定形态学 kernel 缓存，复用临时 buffer，移除热路径日志和不被消费的计算。
- 避免每个候选反复创建整图掩膜；单次连通域/轮廓提取后做候选评分。
- 有界队列优先新帧，明确丢帧策略；记录 camera-to-gaze 的分位延迟。
- 控制 MediaPipe 与 OpenCV 内部线程数，避免过度订阅。

这些是待接入和测量的工程方向，公开演示没有提供相应优化前后性能数据。算法替换、状态拆分与并行应分步验证，避免精度变化与执行顺序变化互相掩盖。
