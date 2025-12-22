// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// 表示一个标签及其关联的十六进制格式颜色。
    /// </summary>
    public record LabelModel
    {
        /// <summary>
        /// 标签索引
        /// </summary>
        public int Index { get; init; }

        /// <summary>
        /// 标签的名称。
        /// </summary>
        public string Name { get; init; } = default!;
    }
}
