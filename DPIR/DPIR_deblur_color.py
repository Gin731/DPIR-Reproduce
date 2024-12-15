import torch
import torch.nn as nn
import torch.fft as fft
import cv2
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        residual = x
        out = self.res(x)
        return out + residual


class DRUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3, num_features=64):
        super(DRUNet, self).__init__()

        # Input convolution layer
        self.m_head = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1, bias=False)

        # Encoder layers with Residual Blocks
        self.m_down1 = nn.Sequential(
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            nn.Conv2d(num_features, num_features * 2, kernel_size=2, stride=2, padding=0, bias=False)
        )

        self.m_down2 = nn.Sequential(
            ResidualBlock(num_features * 2),
            ResidualBlock(num_features * 2),
            ResidualBlock(num_features * 2),
            ResidualBlock(num_features * 2),
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=2, stride=2, padding=0, bias=False)
        )

        self.m_down3 = nn.Sequential(
            ResidualBlock(num_features * 4),
            ResidualBlock(num_features * 4),
            ResidualBlock(num_features * 4),
            ResidualBlock(num_features * 4),
            nn.Conv2d(num_features * 4, num_features * 8, kernel_size=2, stride=2, padding=0, bias=False)
        )

        self.m_body = nn.Sequential(
            ResidualBlock(num_features * 8),
            ResidualBlock(num_features * 8),
            ResidualBlock(num_features * 8),
            ResidualBlock(num_features * 8)
        )

        # Decoder layers with TConv (Transposed Convolution) and Residual Blocks
        self.m_up3 = nn.Sequential(
            nn.ConvTranspose2d(num_features * 8, num_features * 4, kernel_size=2, stride=2, padding=0, bias=False),
            ResidualBlock(num_features * 4),
            ResidualBlock(num_features * 4),
            ResidualBlock(num_features * 4),
            ResidualBlock(num_features * 4)
        )

        self.m_up2 = nn.Sequential(
            nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=2, stride=2, padding=0, bias=False),
            ResidualBlock(num_features * 2),
            ResidualBlock(num_features * 2),
            ResidualBlock(num_features * 2),
            ResidualBlock(num_features * 2)
        )

        self.m_up1 = nn.Sequential(
            nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=2, stride=2, padding=0, bias=False),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            ResidualBlock(num_features),
            ResidualBlock(num_features)
        )

        # Output convolution layer
        self.m_tail = nn.Conv2d(num_features, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x, noise_level):
        print("x shape: ", x.shape)
        noise_level_map = torch.ones_like(x[:, :1, :, :]) * (noise_level/255)
        print("noise_level_map: ", noise_level_map.shape)
        print("Noise level map range:", torch.min(noise_level_map).item(), torch.max(noise_level_map).item())

        x = torch.cat([x,noise_level_map], dim=1)
        print("x shape in NN:" + str(x.shape))

        print("Range after concatenation", torch.min(x).item(), "max =", torch.max(x).item())
        x1 = self.m_head(x)
        print("Range after m_head", torch.min(x1).item(), "max =", torch.max(x1).item())
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)

        # Debug: print shape
        # print("d3 shape:", d3.shape)
        # print("b shape:", b.shape)
        x = self.m_up3(x + x4)

        # print("u3 shape:", u3.shape)
        # print("d2 shape:", d2.shape)
        x = self.m_up2(x + x3)

        # print("u2 shape:", u2.shape)
        # print("d1 shape:", d1.shape)
        x = self.m_up1(x + x2)

        t = self.m_tail(x + x1)
        print("Range after m_tail", torch.min(t).item(), "max =", torch.max(t).item())

        return t


class PlugAndPlayIR:
    def __init__(self, denoiser, T, T_adj, sigma, lam, K=8):
        self.denoiser = denoiser
        self.T = T
        self.T_adj = T_adj
        self.sigma = sigma
        self.lam = lam
        self.K = K

    def data_term_solution(self, y, z, kernel, alpha):
        # Padding to reduce edge artifacts
        pad_height = (y.shape[-2] // 2)
        pad_width = (y.shape[-1] // 2)
        y_padded = torch.nn.functional.pad(y, (pad_width, pad_width, pad_height, pad_height), mode='reflect')
        z_padded = torch.nn.functional.pad(z, (pad_width, pad_width, pad_height, pad_height), mode='reflect')

        # Convert kernel and padded inputs to tensors
        y_padded_tensor, kernel_tensor = y_padded, torch.from_numpy(kernel).float().unsqueeze(0)
        z_padded_tensor = z_padded
        # [kernel_tensor, y_padded_tensor, z_padded_tensor] = [tensor.to(y.device) for tensor in
        #                                                      [kernel_tensor, y_padded_tensor, z_padded_tensor]]

        # Compute FFT of the kernel tensor
        FB = torch.fft.fft2(kernel_tensor, s=(y_padded_tensor.shape[-2], y_padded_tensor.shape[-1]))
        FBC = torch.conj(FB)
        F2B = torch.abs(FB) ** 2
        FBFy = FBC * torch.fft.fft2(y_padded_tensor, dim=(-2, -1))

        print("y padded: "+str(y_padded.shape))
        # Adjust alpha's shape to match y_padded
        alpha_padded = alpha.expand(y_padded.shape[0], y_padded.shape[1], y_padded.shape[2])
        print("alpha padded: "+str(alpha_padded.shape))

        # Compute FFT of the z_padded tensor
        Fz = torch.fft.fft2(z_padded_tensor, dim=(-2, -1))

        # Use the pre-calculated FFT components to solve the data term with padding to reduce edge artifacts
        FR = FBFy + alpha_padded * Fz
        FBR = FR / (F2B + alpha_padded)
        x_padded = torch.real(torch.fft.ifft2(FBR, dim=(-2, -1)))

        # Crop the padded result back to original size
        crop_h_start = pad_height
        crop_h_end = -pad_height if pad_height > 0 else None
        crop_w_start = pad_width
        crop_w_end = -pad_width if pad_width > 0 else None
        x = x_padded[..., crop_h_start:crop_h_end, crop_w_start:crop_w_end]

        return x

    def restore_image(self, y, kernel):
        # initial
        z = y.clone()

        sigma_1 = 49
        sigma_K = self.sigma
        decay_factor = (sigma_K / sigma_1) ** (1 / (self.K - 1))

        for i in range(self.K):
            # Step 1: date term update
            current_sigma = sigma_1 * (decay_factor ** i)
            alpha = lam * (self.sigma ** 2) / (current_sigma ** 2)
            alpha = torch.tensor(alpha, dtype=torch.float32, device=y.device)

            z = self.data_term_solution(y,z, kernel, alpha)
            print("z shape:"+str(z.shape))

            x_image = tensor_to_image_cv2(z)
            cv2.imshow(f"Data Term Solution - Iteration {i + 1}", x_image)
            cv2.waitKey(0)
            cv2.imwrite("Data Term Solution.jpg", x_image)

            # Step 2: prior update
            noise_level = torch.sqrt(torch.tensor(current_sigma)).to(y.device)


            # z = torch.clamp(z, 0, 1)
            print("current noise_level: "+str(noise_level))
            z = self.denoiser(z.unsqueeze(0), noise_level).squeeze(0)
            print("z: " + str(z.shape))

            z_image = tensor_to_image_cv2(z)
            cv2.imshow(f"Denoiser Output - Iteration {i + 1}", z_image)
            cv2.waitKey(0)

        return z



def load_image_cv2(image_path, target_size=256):
    # read image and convert from  BGR to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    h, w, _ = image.shape

    if h == target_size and w == target_size:
        cropped_image = image.astype(np.float32) / 255.0
    else:
        new_h, new_w = min(h, target_size), min(w, target_size)
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        cropped_image = image[start_h:start_h + new_h, start_w:start_w + new_w]
        cropped_image = cv2.resize(cropped_image, (target_size, target_size)).astype(np.float32) / 255.0

    # reorder
    image_tensor = torch.from_numpy(cropped_image).permute(2, 0, 1)  # [3, H, W]
    print("image_tensor: " + str(image_tensor.shape))
    return image_tensor


def tensor_to_image_cv2(tensor):
    tensor = tensor.permute(1, 2, 0)  # convert to [H, W, C]
    print("tensor shape: " + str(tensor.shape))
    # tensor = torch.clamp(tensor, 0, 1)
    image = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)  # remove gradient
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # convert to BGR
    return image

def create_gaussian_kernel(kernel_size=5, sigma=2.0):
    k = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel = k @ k.T
    return gaussian_kernel

def apply_gaussian_blur(image, kernel_size=5, sigma=2.0):
    blurred_images = []
    for c in range(image.shape[0]):
        channel = image[c].cpu().numpy()
        channel_blurred = cv2.GaussianBlur(channel, (kernel_size, kernel_size), sigma)
        blurred_images.append(channel_blurred)

    # combine
    blurred_image = np.stack(blurred_images, axis=0)
    blurred_image_tensor = torch.tensor(blurred_image).float().to(image.device)
    return blurred_image_tensor

def create_expanded_gaussian_kernel(image_size, kernel_size=15, sigma=3.0):
    k = cv2.getGaussianKernel(kernel_size, sigma)
    gaussian_kernel_2d = k @ k.T  # [kernel_size, kernel_size]

    expanded_kernel = np.zeros(image_size, dtype=np.float32)  # [H, W]

    center_y, center_x = image_size[0] // 2, image_size[1] // 2

    start_y = center_y - kernel_size // 2
    start_x = center_x - kernel_size // 2

    expanded_kernel[start_y:start_y + kernel_size, start_x:start_x + kernel_size] = gaussian_kernel_2d

    return expanded_kernel

def add_gaussian_noise(image, mean=0.0, std=0.03):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    return noisy_image



if __name__ == "__main__":

    # Load image
    image_path = "butterfly.png"
    original_image = load_image_cv2(image_path,target_size=256)
    print("original_image: "+ str(original_image.shape))

    cv2.imshow("original_image", tensor_to_image_cv2(original_image))
    cv2.waitKey(0)

    # Add Gaussian blur (simulate degradation)
    blurred_image = apply_gaussian_blur(original_image, kernel_size=5, sigma=2.0)
    print("blurred_image: " + str(blurred_image.shape))

    # # Display the blurred (degraded) image
    # blurred_image_cv2 = tensor_to_image_cv2(blurred_image)
    # cv2.imshow("Blurred Image", blurred_image_cv2)
    # cv2.waitKey(0)
    # cv2.imwrite("Blurred Image.jpg", blurred_image_cv2)

    # Add Gaussian noisy
    noisy_image = add_gaussian_noise(blurred_image)
    print("blurred_image: " + str(noisy_image.shape))

    # Display the noisy image
    noisy_image_cv2 = tensor_to_image_cv2(noisy_image)
    cv2.imshow("Noisy Image", noisy_image_cv2)
    cv2.waitKey(0)
    cv2.imwrite("Noisy Image.jpg", noisy_image_cv2)
    print("Image range before network input: min =", torch.min(noisy_image).item(), "max =",
          torch.max(noisy_image).item())

    # Degenerate Image

    gaussian_kernel = create_gaussian_kernel(kernel_size=5, sigma=2.0)
    T = create_expanded_gaussian_kernel(image_size=(256,256), kernel_size=5, sigma=2.0)
    T_adj = T

    # Algorithm Parameter
    sigma = 25
    lam = 0.23
    K = 8

    # Set denoiser
    denoiser = DRUNet(in_channels=4, out_channels=3)
    weight_path = "drunet_color.pth"
    denoiser.load_state_dict(torch.load(weight_path, weights_only= True))
    denoiser.eval()


    # Restore
    plug_and_play = PlugAndPlayIR(denoiser, T, T_adj, sigma, lam, K)
    restored_image = plug_and_play.restore_image(noisy_image, gaussian_kernel)
    restored_image_cv2 = tensor_to_image_cv2(restored_image)
    cv2.imshow("Restored Image", restored_image_cv2)
    cv2.waitKey(0)
    cv2.imwrite("restored_image_cv2.jpg", restored_image_cv2)
