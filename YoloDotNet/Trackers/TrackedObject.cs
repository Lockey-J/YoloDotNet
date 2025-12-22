// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    public class TrackedObject(IDetection boundingBox, int tailLength = 30)
    {
        public IDetection BoundingBox { get; private set; } = boundingBox;
        public int Age { get; set; } = 0;
        public KalmanFilter Kalman { get; private set; } = new KalmanFilter(boundingBox.BoundingBox.MidX, boundingBox.BoundingBox.MidY);

        private readonly TailTrack _tailTracker = new(tailLength);

        public void TrackBoundingBox(IDetection detection)
        {
            // 更新边界框
            BoundingBox = detection;

            // 存储当前边界框中心坐标
            var box = detection.BoundingBox;

            // 更新轨迹
            detection.Tail = _tailTracker.GetTail();

            _tailTracker.AddTailPoint(new SKPointI(box.MidX, box.MidY)); // 存储边界框的中心。

            // 更新卡尔曼滤波器
            Kalman.Update(box.MidX, box.MidY);
        }

        public void KalmanPredict()
            => Kalman.Predict();

        public float[] GetPredictedState()
            => Kalman.GetState();
    }
}
