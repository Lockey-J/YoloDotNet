// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Video
{
    /// <summary>
    /// 配置视频处理的选项。
    /// </summary>
    public class VideoOptions
    {
        /// <summary>
        /// 要处理的视频文件路径或实时视频流的 URL。
        /// </summary>
        public string VideoInput { get; set; } = default!;

        /// <summary>
        /// 可选：输出视频文件的路径，处理后的视频将保存在此。
        /// </summary>
        public string VideoOutput{ get; set; } = default!;

        /// <summary>
        /// 写入输出时使用的编码器。
        /// 仅在设置了 <see cref="VideoOutput"/> 时相关。
        /// </summary>
        public VideoEncoder VideoEncoder { get; set; }

        /// <summary>
        /// 获取或设置输出视频的宽度。
        /// 默认值为 0，表示保留源宽度。
        /// 设置为 -2 可根据指定的高度自动计算宽度，
        /// 同时保持宽高比。
        /// 注意：宽度和高度一次只能有一个设置为 -2。
        /// </summary>
        public int Width { get; set; }

        /// <summary>
        /// 获取或设置输出视频的高度。
        /// 默认值为 0，表示保留源高度。
        /// 设置为 -2 可根据指定的宽度自动计算高度，
        /// 同时保持宽高比。
        /// 注意：宽度和高度一次只能有一个设置为 -2。
        /// </summary>
        public int Height { get; set; }

        /// <summary>
        /// 设置输出视频的每秒帧数 (FPS)。
        /// </summary>
        public FrameRate FrameRate { get; set; } = FrameRate.AUTO;

        /// <summary>
        /// 获取或设置处理帧的间隔。
        /// 只有第 n 帧将被处理，其中 n 是此属性的值。
        /// 对于不需要处理每一帧的监控或监视场景很有用。
        /// </summary>
        public int FrameInterval { get; set; }

        /// <summary>
        /// 每个视频片段的持续时间（秒）。视频将被分割为该长度的块。
        /// 默认值为 600（10 分钟）。
        /// </summary>
        public int VideoChunkDuration { get; set; } = 600;

        /// <summary>
        /// 输出视频的压缩质量（0–51）。数值越低质量越好。默认值为 30。
        /// </summary>
        public int CompressionQuality { get; set; } = 30;

        /// <summary>
        /// 开始处理视频的起始时间（秒）。
        /// 仅在输入是视频文件（而非实时流）时相关。
        /// </summary>
        public float StartTimeSeconds { get; set; }
        
        /// <summary>
        /// 处理视频的持续时间（秒）。
        /// 仅在输入是视频文件（而非实时流）时相关。
        /// </summary>
        public float DurationSeconds { get; set; }
     
    }
}
