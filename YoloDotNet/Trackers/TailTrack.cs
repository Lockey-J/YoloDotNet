// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    /// <summary>
    /// 存储固定长度的轨迹点队列，用于跟踪和可视化对象的运动轨迹。
    /// </summary>
    /// <param name="maxLength"></param>
    public class TailTrack(int maxLength)
    {
        private readonly int _maxLength = maxLength;
        private readonly Queue<SKPoint> _positions = new (maxLength);

        /// <summary>
        /// 返回当前轨迹点的副本作为列表。
        /// </summary>
        public List<SKPoint> GetTail() => [.. _positions];

        /// <summary>
        /// 向轨迹添加新点。
        /// 如果轨迹超过最大长度，则自动移除最旧的点。
        /// </summary>
        public void AddTailPoint(SKPoint point)
        {
            if (_positions.Count >= _maxLength)
                _positions.Dequeue();

            _positions.Enqueue(point);
        }
    }
}
