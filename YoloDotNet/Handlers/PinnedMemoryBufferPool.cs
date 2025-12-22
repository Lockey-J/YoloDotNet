// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Handlers
{
    /// <summary>
    /// 用于管理由 SKBitmap 实例支持的可重用固定内存缓冲区的池。
    /// 有助于减少高频图像处理场景中的 GC 压力和分配成本。
    /// </summary>
    public class PinnedMemoryBufferPool : IDisposable
    {
        // 可重用固定内存缓冲区的内部线程安全池
        internal readonly ConcurrentBag<PinnedMemoryBuffer> _pool = [];

        // 用于分配每个 SKBitmap 的图像格式/维度
        private readonly SKImageInfo _imageInfo;

        /// <summary>
        /// 使用指定的图像布局初始化缓冲池并预分配多个缓冲区。
        /// </summary>
        public PinnedMemoryBufferPool(SKImageInfo skInfo, int initialSize = 60)
        {
            _imageInfo = skInfo;

            for (int i = 0; i < initialSize; i++)
                _pool.Add(new PinnedMemoryBuffer(_imageInfo));
        }

        /// <summary>
        /// 从池中检索缓冲区，如果池为空则创建新的缓冲区。
        /// </summary>
        public PinnedMemoryBuffer Rent()
        {
            if (_pool.TryTake(out var buffer))
                return buffer;

            // 池已耗尽 — 创建新缓冲区作为后备
            return new PinnedMemoryBuffer(_imageInfo); // fallback
        }

        /// <summary>
        /// 在清除缓冲区内容后，将使用过的缓冲区返回到池中。
        /// </summary>
        /// <param name="buffer">要返回和重用的缓冲区。</param>
        public void Return(PinnedMemoryBuffer buffer)
        {
            // 重要：在重用前清除位图以防止视觉伪影。
            // 这避免了将旧帧数据泄漏到后续帧中。
            // Using SKColors.Empty fills with transparent black (0,0,0,0).
            buffer.TargetBitmap.Erase(SKColors.Empty);

            _pool.Add(buffer);
        }

        /// <summary>
        /// 释放池使用的所有资源。
        /// </summary>
        public void Dispose()
        {
            while (_pool.TryTake(out var buffer))
                buffer?.Dispose();

            GC.SuppressFinalize(this);
        }
    }
}
