// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Configuration
{
    internal static class SystemPlatform
    {
        /// <summary>
        /// 检测当前操作系统平台。
        /// </summary>
        /// <returns>表示检测到的操作系统的 OSPlatformType 枚举。</returns>
        public static Platform GetOS()
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
                return Platform.Linux;

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return Platform.Windows;

            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return Platform.MacOS;

            return Platform.Unknown;
        }
    }
}
