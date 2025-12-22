// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Video
{
    public readonly struct FrameRate
    {
        public float Value { get; }

        private FrameRate(float value)
        {
            Value = value;
        }

        // 预设
        public static readonly FrameRate AUTO = new(0);
        public static readonly FrameRate FPS15 = new (15f);
        public static readonly FrameRate FPS23_976 = new (23.976f);
        public static readonly FrameRate FPS24 = new (24f);
        public static readonly FrameRate FPS25 = new (25f);
        public static readonly FrameRate FPS29_97 = new (29.97f);
        public static readonly FrameRate FPS30 = new (30f);
        public static readonly FrameRate FPS50 = new (50f);
        public static readonly FrameRate FPS59_94 = new (59.94f);
        public static readonly FrameRate FPS60 = new (60f);

        // 为灵活性而进行的隐式转换
        public static implicit operator float(FrameRate rate) => rate.Value;
    }
}
