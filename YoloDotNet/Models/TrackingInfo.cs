// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// 包含跟踪相关的元数据，如分配的 ID 和运动轨迹，
    /// 仅在启用对象跟踪时可用。
    /// </summary>
    [EditorBrowsable(EditorBrowsableState.Never)]
    public class TrackingInfo
    {
        /// <summary>
        /// 跟踪器分配给对象的唯一标识符（如果启用跟踪）。
        /// </summary>
        public int? Id { get; set; }

        /// <summary>
        /// 表示跟踪对象的轨迹或路径历史记录的点列表，
        /// 用于可视化跨帧的运动。
        /// </summary>
        public List<SKPoint>? Tail { get; set; }
    }
}
