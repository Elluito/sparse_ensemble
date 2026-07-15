# PyTorch adaptation of the TensorFlow spectral pooling implementation from
# https://github.com/VijayKalmath/Spectral-Representations-for-Convolutional-Neural-Networks/blob/master/src/modules/spectral_pooling.py
#
# Spectral pooling (Rippel et al., 2015) downsamples a feature map by
# transforming it to the frequency domain, cropping/masking around the DC
# component, and transforming back to the spatial domain.
import math

import torch
import torch.nn as nn


class SpectralPooling(nn.Module):
    """
    Spectral pooling layer operating on channels-first tensors [N, C, H, W].

    Two mutually exclusive modes, mirroring the original ``spectral_pool`` and
    ``utils_spectral_pool`` methods:

    - ``filter_shape=(h, w)``: crops the (shifted) Fourier transform down to
      an ``h x w`` window around the DC component, actually reducing the
      spatial resolution of the output.
    - ``pool_size=k``: keeps the original spatial resolution but zeroes out
      all frequencies outside the central ``1/k`` band (a low-pass mask
      rather than a crop).

    Unlike the reference implementation, ``fftshift``/``ifftshift`` are
    applied only over the spatial dimensions (H, W); the original TF code
    shifts over every axis by default, which also permutes the batch and
    channel dimensions and only happened to be harmless there because it was
    exercised with batch size 1.
    """

    def __init__(self, filter_shape=None, pool_size=None, normalize=True):
        super().__init__()
        if (filter_shape is None) == (pool_size is None):
            raise ValueError("Provide exactly one of `filter_shape` or `pool_size`")

        self.filter_shape = tuple(filter_shape) if filter_shape is not None else None
        self.pool_size = pool_size
        self.normalize = normalize

    @staticmethod
    def fourier_transform(spatial_image):
        """Real/complex [N, C, H, W] -> complex [N, C, H, W], DC shifted to center."""
        fourier_image = torch.fft.fft2(spatial_image)
        fourier_image = torch.fft.fftshift(fourier_image, dim=(-2, -1))
        return fourier_image

    @staticmethod
    def inverse_fourier_transform(fourier_image):
        """Complex [N, C, H, W], DC centered -> complex [N, C, H, W]."""
        fourier_image = torch.fft.ifftshift(fourier_image, dim=(-2, -1))
        spatial_image = torch.fft.ifft2(fourier_image)
        return spatial_image

    @staticmethod
    def normalize_image(spatial_image):
        """Min-max normalize each image (and channel) independently to [0, 1].

        Expects channels-first [N, C, H, W]; reduces over the spatial dims.
        """
        channel_max = torch.amax(spatial_image, dim=(-2, -1), keepdim=True)
        channel_min = torch.amin(spatial_image, dim=(-2, -1), keepdim=True)
        return (spatial_image - channel_min) / (channel_max - channel_min)

    @staticmethod
    def lowpass_filter_crop(fourier_image, filter_shape):
        """Crop the Fourier image to `filter_shape` around its center."""
        filter_shape = list(filter_shape)
        height, width = fourier_image.shape[-2], fourier_image.shape[-1]

        if (height + filter_shape[0]) % 2 == 1:
            filter_shape[0] += 1
        if (width + filter_shape[1]) % 2 == 1:
            filter_shape[1] += 1

        top = (height - filter_shape[0]) // 2
        bottom = top + filter_shape[0]
        left = (width - filter_shape[1]) // 2
        right = left + filter_shape[1]

        return fourier_image[..., top:bottom, left:right]

    @staticmethod
    def lowpass_filter_poolsize(fourier_image, pool_size):
        """Zero out all frequencies outside the central 1/pool_size band."""
        height, width = fourier_image.shape[-2], fourier_image.shape[-1]
        mask = torch.ones_like(fourier_image.real)

        dist_h = math.ceil((height - height / pool_size) / 2)
        if dist_h > 0:
            mask[..., :dist_h, :] = 0
            mask[..., -dist_h:, :] = 0

        dist_w = math.ceil((width - width / pool_size) / 2)
        if dist_w > 0:
            mask[..., :, :dist_w] = 0
            mask[..., :, -dist_w:] = 0

        return fourier_image * mask

    @staticmethod
    def treat_corner_cases(filtered_fourier_image):
        """Zero the imaginary part of self-conjugate (unpaired Nyquist) frequency bins."""
        height = filtered_fourier_image.shape[-2]
        width = filtered_fourier_image.shape[-1]

        corner_set = [(0, 0)]
        if height % 2 == 0:
            corner_set.append((height // 2, 0))
            if width % 2 == 0:
                corner_set.append((height // 2, width // 2))
        if width % 2 == 0:
            corner_set.append((0, width // 2))

        mask = torch.ones(height, width, device=filtered_fourier_image.device)
        for i, j in corner_set:
            mask[i, j] = 0

        real = filtered_fourier_image.real
        imag = filtered_fourier_image.imag * mask
        return torch.complex(real, imag)

    def forward(self, spatial_image):
        """
        spatial_image: real-valued tensor [N, C, H, W]

        Returns:
            magnitude_spectrum: [N, C, H', W'] float tensor, log-magnitude of
                the (corner-treated) filtered spectrum, normalized to [0, 1].
            filtered_spatial_image: [N, C, H', W'] float tensor, the pooled
                feature map back in the spatial domain.
        """
        fourier_image = self.fourier_transform(spatial_image.to(torch.complex64))

        if self.filter_shape is not None:
            if self.filter_shape[0] > 1:
                filtered_fourier_image = self.lowpass_filter_crop(fourier_image, self.filter_shape)
            else:
                filtered_fourier_image = fourier_image
        else:
            if self.pool_size > 1:
                filtered_fourier_image = self.lowpass_filter_poolsize(fourier_image, self.pool_size)
            else:
                filtered_fourier_image = fourier_image

        filtered_fourier_image = self.treat_corner_cases(filtered_fourier_image)

        filtered_spatial_image = torch.abs(self.inverse_fourier_transform(filtered_fourier_image))
        if self.normalize:
            filtered_spatial_image = self.normalize_image(filtered_spatial_image)

        magnitude_spectrum = 20 * torch.log(torch.abs(filtered_fourier_image) + 1e-45)
        magnitude_spectrum = magnitude_spectrum / magnitude_spectrum.amax()

        return magnitude_spectrum, filtered_spatial_image


class SpectralMaxPool2d(nn.Module):
    """
    Drop-in replacement for ``nn.MaxPool2d`` that reduces the spatial size of
    its input by the same amount ``nn.MaxPool2d(kernel_size, stride,
    padding)`` would, but performs the reduction via ``SpectralPooling``'s
    frequency-domain crop instead of a spatial max.

    Same constructor signature as ``nn.MaxPool2d`` (only square/int
    kernel_size, stride, padding, dilation are supported, matching how this
    codebase uses it). The target output shape is computed from the input's
    actual (H, W) with the standard pooling output-size formula, so it also
    works for inputs other than the 32x32 CIFAR tensors this was built for.

    Note: ``SpectralPooling.lowpass_filter_crop`` may grow the requested crop
    by one pixel to keep the crop symmetric around the DC component when its
    parity doesn't match the input's, so the output can be up to one pixel
    larger than what ``nn.MaxPool2d`` would produce. This is harmless for the
    conv/batchnorm/adaptive-pool layers that follow it in these models.
    """

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1, normalize=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.normalize = normalize

    @staticmethod
    def _output_size(size, kernel_size, stride, padding, dilation):
        return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

    def forward(self, x):
        height, width = x.shape[-2], x.shape[-1]
        out_h = self._output_size(height, self.kernel_size, self.stride, self.padding, self.dilation)
        out_w = self._output_size(width, self.kernel_size, self.stride, self.padding, self.dilation)

        fourier_image = SpectralPooling.fourier_transform(x.to(torch.complex64))
        filtered_fourier_image = SpectralPooling.lowpass_filter_crop(fourier_image, (out_h, out_w))
        filtered_fourier_image = SpectralPooling.treat_corner_cases(filtered_fourier_image)

        filtered_spatial_image = torch.abs(SpectralPooling.inverse_fourier_transform(filtered_fourier_image))
        if self.normalize:
            filtered_spatial_image = SpectralPooling.normalize_image(filtered_spatial_image)

        return filtered_spatial_image
