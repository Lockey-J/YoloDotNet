// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    public static class CommonExtension
    {
        /// <summary>
        /// 将值转换为带两位小数的百分比的字符串表示。
        /// </summary>
        /// <param name="value">要转换的值。</param>
        /// <returns>带两位小数的百分比字符串表示。</returns>
        public static string ToPercent(this double value)
            => (value * 100).ToString("0.##", CultureInfo.InvariantCulture);

        /// <summary>
        /// 根据图像维度和给定缩放值计算新的动态大小。
        /// </summary>
        /// <param name="image">要计算动态大小的图像。</param>
        /// <param name="scale">要动态调整的初始缩放值。</param>
        /// <returns>作为浮点值的新的动态计算大小。</returns>
        public static float CalculateDynamicSize(this SKBitmap image, float scale)
        {
            // 根据图像分辨率和分母计算缩放因子
            float scaleFactor = image.Width / ImageConfig.SCALING_DENOMINATOR;

            var newSize = scale;

            newSize *= scaleFactor;

            return Math.Max(newSize, scale);
        }

        /// <summary>
        /// 过滤对象检测结果列表，仅保留标签与指定过滤器类匹配的对象。
        /// </summary>
        /// <typeparam name="T">检测结果的类型。必须是受支持的类型之一。</typeparam>
        /// <param name="result">要过滤的检测结果列表。</param>
        /// <param name="filterClasses">要在过滤结果中保留的类标签集合。</param>
        /// <returns>仅包含标签与任何指定过滤器类匹配的检测结果的过滤列表。</returns>
        /// <exception cref="ArgumentException">如果类型 <typeparamref name="T"/> 不是受支持的检测类型，则抛出。</exception>
        public static List<T> FilterLabels<T>(this IEnumerable<T> result, HashSet<string> filterClasses) where T : IDetection
        {
            ArgumentNullException.ThrowIfNull(result);
            ArgumentNullException.ThrowIfNull(filterClasses);

            var filtered = new List<T>();

            foreach (var detection in result)
            {
                var label = detection.Label.Name ?? detection.Label.ToString();

                if (filterClasses.Contains(label))
                    filtered.Add(detection);
            }

            return filtered;
        }
    }
}
