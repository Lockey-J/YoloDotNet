// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Attributes
{
    /// <summary>
    /// 用于指定与枚举值关联的编码器名称的属性。
    /// </summary>
    /// <param name="name"></param>
    [AttributeUsage(AttributeTargets.Field)]
    public sealed class EncoderNameAttribute(string name) : Attribute
    {
        public string Name { get; } = name;
    }
}
