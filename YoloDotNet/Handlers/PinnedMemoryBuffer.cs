// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Handlers
{
    public class PinnedMemoryBuffer : IDisposable
    {
        public readonly SKImageInfo ImageInfo;
        public readonly byte[] Buffer;
        public readonly IntPtr Pointer;
        public readonly SKBitmap TargetBitmap;
        public readonly SKCanvas Canvas;

        private readonly GCHandle _handle;

        public PinnedMemoryBuffer(SKImageInfo imageInfo)
        {
            ImageInfo = imageInfo;

            //var _imageInfo = new SKImageInfo(width, height, SKColorType.Rgb888x, SKAlphaType.Opaque);
            Buffer = new byte[imageInfo.BytesSize];

            _handle = GCHandle.Alloc(Buffer, GCHandleType.Pinned);
            Pointer = _handle.AddrOfPinnedObject();

            // 将固定缓冲区包装在SKBitmap中，以便我们可以在其中绘制
            TargetBitmap = new SKBitmap();

            if (!TargetBitmap.InstallPixels(imageInfo, Pointer, imageInfo.RowBytes))
                throw new YoloDotNetException("Failed to install pixels into SKBitmap");

            Canvas = new SKCanvas(TargetBitmap);
        }

        public void Dispose()
        {
            Canvas?.Dispose();
            TargetBitmap?.Dispose();

            if (_handle.IsAllocated)
                _handle.Free();

            GC.SuppressFinalize(this);
        }
    }
}
