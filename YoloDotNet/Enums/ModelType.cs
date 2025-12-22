// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Enums
{
    /// <summary>
    /// 图像视觉类型的强类型名称。
    /// </summary>
    [DataContract]
    public enum ModelType
    {
        [EnumMember(Value = "classify")]
        Classification,

        [EnumMember(Value = "detect")]
        ObjectDetection,

        [EnumMember(Value = "obb")]
        ObbDetection,

        [EnumMember(Value = "segment")]
        Segmentation,

        [EnumMember(Value = "pose")]
        PoseEstimation
    }
}
