// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (c) 2023-2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Extensions
{
    public static class ImageResizeExtension
    {
        /// <summary>
        /// 通过拉伸输入图像以适应模型输入大小来调整图像尺寸，返回指向 RGB888x 像素数据的指针和新尺寸。
        /// </summary>
        /// <param name="img">要调整大小的原始图像。</param>
        /// <param name="samplingOptions">调整大小时使用的采样选项。</param>
        /// <param name="pinnedMemoryBuffer">将写入调整大小图像的固定内存缓冲区。</param>
        /// <returns>包含指向调整大小图像数据的指针及其尺寸的元组。</returns>
        public static SKSizeI ResizeImageStretched<T>(this T img, SKSamplingOptions samplingOptions, PinnedMemoryBuffer pinnedMemoryBuffer)
        {
            SKImage image = default!;
            var createdImage = false;

            if (img is SKImage skImage)
                image = skImage;
            else if (img is SKBitmap skBitmap)
            {
                image = SKImage.FromPixels(skBitmap.Info, skBitmap.GetPixels());
                createdImage = true;
            }

            int modelWidth = pinnedMemoryBuffer.ImageInfo.Width;
            int modelHeight = pinnedMemoryBuffer.ImageInfo.Height;

            var srcRect = new SKRect(0, 0, image.Width, image.Height);
            var destRect = new SKRect(0, 0, modelWidth, modelHeight);

            pinnedMemoryBuffer.Canvas.DrawImage(image, srcRect, destRect, samplingOptions);
            var w = image.Width;
            var h = image.Height;

            // 只有当我们从 SKBitmap 创建了新的 SKImage 时才释放
            if (createdImage)
                image?.Dispose();

            // 返回原始图像尺寸，这是正确缩放边界框所需的
            return new SKSizeI(w, h);
        }

        /// <summary>
        /// 按比例调整输入图像大小以适应模型输入尺寸，使用 RGB888x 格式和填充边框，返回指向像素数据的指针和新图像尺寸。
        /// </summary>
        /// <param name="img">要调整大小的原始图像。</param>
        /// <param name="samplingOptions">调整大小时使用的采样选项。</param>
        /// <param name="pinnedMemoryBuffer">将写入调整大小图像的固定内存缓冲区。</param>
        /// <returns>包含指向调整大小图像数据的指针及其尺寸的元组。</returns>
        public static SKSizeI ResizeImageProportional<T>(this T img, SKSamplingOptions samplingOptions, PinnedMemoryBuffer pinnedMemoryBuffer)
        {
            SKImage image = default!;
            var createdImage = false;

            if (img is SKImage skImage)
                image = skImage;
            else if (img is SKBitmap skBitmap)
            {
                image = SKImage.FromPixels(skBitmap.Info, skBitmap.GetPixels());
                createdImage = true;
            }

            int modelWidth = pinnedMemoryBuffer.ImageInfo.Width;
            int modelHeight = pinnedMemoryBuffer.ImageInfo.Height;
            int width = image.Width;
            int height = image.Height;

            // 根据宽高比计算新的图像尺寸
            float scaleFactor = Math.Min((float)modelWidth / width, (float)modelHeight / height);

            // 使用整数舍入而不是 Math.Round
            int newWidth = (int)((width * scaleFactor) + 0.5f);
            int newHeight = (int)((height * scaleFactor) + 0.5f);

            // 计算模型尺寸内的目标矩形
            int x = (modelWidth - newWidth) / 2;
            int y = (modelHeight - newHeight) / 2;

            var srcRect = new SKRect(0, 0, width, height);
            var dstRect = new SKRect(x, y, x + newWidth, y + newHeight);

            pinnedMemoryBuffer.Canvas.DrawImage(image, srcRect, dstRect, samplingOptions);
            var w = image.Width;
            var h = image.Height;

            // 只有当我们从 SKBitmap 创建了新的 SKImage 时才释放
            if (createdImage)
                image?.Dispose();

            // 返回原始图像尺寸，这是正确缩放边界框所需的
            return new SKSizeI(w, h);
        }

        /// <summary>
        /// 将原始像素图像数据转换为归一化的浮点数组用于模型输入。
        /// </summary>
        /// <param name="pixelsPtr">指向内存中原始像素图像数据的指针。</param>
        /// <param name="inputShape">输入张量的形状。</param>
        /// <param name="tensorBufferSize">张量缓冲区的大小，应等于输入形状维度的乘积。</param>
        /// <param name="tensorArrayBuffer">用于存储归一化像素值的预分配浮点数组缓冲区。</param>
        unsafe public static void NormalizePixelsToArray(this IntPtr pixelsPtr,
            long[] inputShape,
            int tensorBufferSize,
            float[] tensorArrayBuffer)
        {
            var colorChannels = (int)inputShape[1];
            var height = (int)inputShape[2];
            var width = (int)inputShape[3];
            int totalPixels = width * height;

            float inv255 = 1.0f / 255.0f;
            byte* src = (byte*)pixelsPtr;

            if (colorChannels == 1)
            {
                float* dst = (float*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);
                int srcIndex = 0;

                for (int i = 0; i < totalPixels; i++, srcIndex += 4)
                {
                    // 只读取灰度分量（假设在 R 通道中）
                    dst[i] = src[srcIndex] * inv255;
                }
            }
            else
            {
                float* dstR = (float*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);
                float* dstG = dstR + totalPixels;
                float* dstB = dstG + totalPixels;

                int srcIndex = 0;
                for (int i = 0; i < totalPixels; i++, srcIndex += 4)
                {
                    dstR[i] = src[srcIndex] * inv255;
                    dstG[i] = src[srcIndex + 1] * inv255;
                    dstB[i] = src[srcIndex + 2] * inv255;
                }
            }
        }

        /// <summary>
        /// NormalizePixelsToArray 的重载，将原始像素图像数据转换为归一化的半精度浮点（ushort）数组用于模型输入。
        /// </summary>
        /// <param name="pixelsPtr">指向内存中原始像素图像数据的指针。</param>
        /// <param name="inputShape">输入张量的形状。</param>
        /// <param name="tensorBufferSize">张量缓冲区的大小。</param>
        /// <param name="tensorArrayBuffer">用于存储归一化像素值的预分配数组缓冲区。</param>
        unsafe public static void NormalizePixelsToArray(this IntPtr pixelsPtr,
            long[] inputShape,
            int tensorBufferSize,
            ushort[] tensorArrayBuffer)
        {
            var colorChannels = (int)inputShape[1];
            var height = (int)inputShape[2];
            var width = (int)inputShape[3];
            int totalPixels = width * height;

            float inv255 = 1.0f / 255.0f;
            byte* src = (byte*)pixelsPtr;

            if (colorChannels == 1)
            {
                ushort* dst = (ushort*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);
                int srcIndex = 0;

                for (int i = 0; i < totalPixels; i++, srcIndex += 4)
                {
                    dst[i] = FloatToUshort(src[srcIndex] * inv255);
                }
            }
            else
            {
                ushort* dstR = (ushort*)Unsafe.AsPointer(ref tensorArrayBuffer[0]);
                ushort* dstG = dstR + totalPixels;
                ushort* dstB = dstG + totalPixels;

                int srcIndex = 0;
                for (int i = 0; i < totalPixels; i++, srcIndex += 4)
                {
                    dstR[i] = FloatToUshort(src[srcIndex] * inv255);
                    dstG[i] = FloatToUshort(src[srcIndex + 1] * inv255);
                    dstB[i] = FloatToUshort(src[srcIndex + 2] * inv255);
                }
            }
        }

        // 将浮点数转换为半精度（16位）浮点数（ushort）的辅助方法
        unsafe private static ushort FloatToUshort(float value)
        {
            // 出于性能原因避免使用 BitConverter，而是使用不安全转换。
            uint f = *(uint*)&value;

            // 提取部分
            int sign = (int)(f >> 16) & 0x8000;
            int exponent = (int)((f >> 23) & 0xFF) - 112;
            int mantissa = (int)(f & 0x7FFFFF);

            if (exponent <= 0)
            {
                if (exponent < -10)
                {
                    return (ushort)sign; // 太小 -> 零
                }
                mantissa = (mantissa | 0x800000) >> (1 - exponent);
                return (ushort)(sign | (mantissa + 0xFFF + ((mantissa >> 13) & 1)) >> 13);
            }
            else if (exponent == 143 - 112) // 无穷大/NaN
            {
                if (mantissa == 0)
                    return (ushort)(sign | 0x7C00); // 无穷大
                return (ushort)(sign | 0x7C00 | (mantissa >> 13)); // NaN
            }
            else
            {
                if (exponent > 30)
                {
                    return (ushort)(sign | 0x7C00); // 溢出 -> 无穷大
                }
                return (ushort)(sign | (exponent << 10) | (mantissa + 0xFFF + ((mantissa >> 13) & 1)) >> 13);
            }
        }
    }
}
