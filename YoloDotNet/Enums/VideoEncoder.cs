// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Enums
{
    /// <summary>
    /// 通用视频编码器。
    /// 注意：FFmpeg 必须构建为支持选定的编码器。
    /// </summary>
    public enum VideoEncoder
    {
        #region H.264 / AVC

        /// <summary>
        /// 软件 H.264 编码器（CPU）。
        /// 通用支持，质量好，性能较慢。
        /// </summary>
        [EncoderName("libx264")]
        LibX264,

        /// <summary>
        /// NVIDIA NVENC H.264 (GPU)。
        /// 快速硬件编码器。需要 NVIDIA GPU 和构建了 NVENC 的 FFmpeg。
        /// </summary>
        [EncoderName("h264_nvenc")]
        H264Nvenc,

        /// <summary>
        /// Intel Quick Sync H.264 (GPU)。
        /// 需要 Intel iGPU 和构建了 QSV 支持的 FFmpeg。
        /// </summary>
        [EncoderName("h264_qsv")]
        H264Qsv,

        /// <summary>
        /// AMD AMF H.264 (GPU)。
        /// 需要 AMD GPU 和构建了 AMF 的 FFmpeg。
        /// </summary>
        [EncoderName("h264_amf")]
        H264Amf,

        /// <summary>
        /// VAAPI H.264（仅限 Linux）。
        /// 适用于 Intel/AMD GPU 的通用硬件编码器。
        /// </summary>
        [EncoderName("h264_vaapi")]
        H264Vaapi,

        /// <summary>
        /// Apple VideoToolbox H.264 (macOS).
        /// 在 Apple 设备上进行硬件加速编码。
        /// </summary>
        [EncoderName("h264_videotoolbox")]
        H264VideoToolbox,

        #endregion

        #region H.265 / HEVC

        /// <summary>
        /// 软件 H.265/HEVC 编码器（CPU）。
        /// 压缩效率非常高，但相比硬件编码器速度较慢。
        /// </summary>
        [EncoderName("libx265")]
        LibX265,

        /// <summary>
        /// NVIDIA NVENC HEVC (GPU).
        /// 需要 NVIDIA GPU 和构建了 NVENC 的 FFmpeg。
        /// </summary>
        [EncoderName("hevc_nvenc")]
        HevcNvenc,

        /// <summary>
        /// Intel Quick Sync HEVC (GPU).
        /// 需要 Intel iGPU 和构建了 QSV 支持的 FFmpeg。
        /// </summary>
        [EncoderName("hevc_qsv")]
        HevcQsv,

        /// <summary>
        /// AMD AMF HEVC (GPU).
        /// 需要 AMD GPU 和构建了 AMF 的 FFmpeg。
        /// </summary>
        [EncoderName("hevc_amf")]
        HevcAmf,

        /// <summary>
        /// VAAPI HEVC（仅限 Linux）。
        /// 适用于 Intel/AMD GPU 的通用硬件编码器。
        /// </summary>
        [EncoderName("hevc_vaapi")]
        HevcVaapi,

        /// <summary>
        /// Apple VideoToolbox HEVC (macOS).
        /// 在 Apple 设备上进行硬件加速 HEVC 编码。
        /// </summary>
        [EncoderName("hevc_videotoolbox")]
        HevcVideoToolbox,

        #endregion

        #region AV1

        /// <summary>
        /// 软件 AV1 编码器（CPU）。
        /// 压缩效率非常高，但速度极慢。
        /// </summary>
        [EncoderName("libaom-av1")]
        LibAomAv1,

        /// <summary>
        /// NVIDIA NVENC AV1 (GPU)。
        /// 需要 NVIDIA Ampere/Ada GPU（RTX 30/40 系列）和构建了 NVENC 的 FFmpeg。
        /// </summary>
        [EncoderName("av1_nvenc")]
        Av1Nvenc,

        /// <summary>
        /// Intel Quick Sync AV1 (GPU)。
        /// 需要 Intel Arc GPU 或近期的 iGPU，以及构建了 QSV 的 FFmpeg。
        /// </summary>
        [EncoderName("av1_qsv")]
        Av1Qsv,

        /// <summary>
        /// VAAPI AV1（仅限 Linux）。
        /// 适用于 Intel/AMD GPU 的通用硬件编码器。
        /// </summary>
        [EncoderName("av1_vaapi")]
        Av1Vaapi,

        /// <summary>
        /// Apple VideoToolbox AV1（macOS 14+）。
        /// 在 Apple Silicon 上进行硬件加速的 AV1 编码。
        /// </summary>
        [EncoderName("av1_videotoolbox")]
        Av1VideoToolbox,

        #endregion

        #region ProRes

        /// <summary>
        /// 通过 VideoToolbox 的 Apple ProRes 编码器（macOS）。
        /// 高质量的帧内编码器，用于编辑工作流程。
        /// </summary>
        [EncoderName("prores_videotoolbox")]
        ProResVideoToolbox

        #endregion
    }
}
