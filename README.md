### **README: DRUNet-Based Plug-and-Play Image Restoration**

---

### **Project Description**
This repository provides a **Plug-and-Play Image Restoration** implementation using the **DRUNet denoiser**. The method is based on integrating denoisers as a prior for solving image restoration tasks such as **deblurring** and **denoising**.

The code implements the algorithm described in the **Denoising Prior Driven Image Restoration (DPIR)** method, utilizing a pre-trained **DRUNet** model and Plug-and-Play ADMM techniques.

---

### **Features**
- **Image Restoration**: Handles Gaussian blur degradation and additive Gaussian noise.
- **DRUNet Integration**: Implements a deep residual U-Net denoiser (DRUNet) as a prior.
- **Plug-and-Play Framework**: Integrates DRUNet into an iterative restoration framework.
- **Flexible Parameters**: Customizable kernel size, noise level, iterations.

---

### **Requirements**
Install the necessary libraries before running the code:

```bash
pip install torch torchvision opencv-python numpy
```

---


### **How to Run**

1. **Set Up the Environment**  
   Install the dependencies:
   ```bash
   pip install torch torchvision opencv-python numpy
   ```

2. **Prepare Input Image**  
   Place an input image (e.g., `butterfly.png`) in the project directory.

3. **Run the Script**  
   Execute the main script:
   ```bash
   python main.py
   ```

4. **Results**  
   - The program outputs intermediate results for each iteration:
     - **Degraded Image** (Blurred + Noisy)
     - **Restored Image** after processing with the DRUNet denoiser.

---

### **Parameters**
You can customize the following parameters in the script:

- **Noise Level**:  
   Set `sigma` to control the noise standard deviation:
   ```python
   sigma = 25
   ```

- **Gaussian Kernel**:  
   Modify kernel size and standard deviation:
   ```python
   kernel_size = 5
   sigma_kernel = 2.0
   ```

- **Plug-and-Play ADMM Settings**:  
   Set the number of iterations `K` and regularization parameter `lam`:
   ```python
   lam = 0.23
   K = 8
   ```

---

### **Input & Output Example**

#### Input:
Original input image: `butterfly.png`  
Degraded image: Gaussian blur + Gaussian noise.

#### Output:
- **Noisy Image**  
   ![Noisy Image](./Noisy%20Image.jpg)
- **Restored Image**  
   ![Restored Image](./restored_image_cv2.jpg)

---

### **Model Details**
- **DRUNet**: A deep residual U-Net denoiser with pre-trained weights (`drunet_color.pth`).
- **Plug-and-Play**: ADMM framework iteratively solves the data fidelity term and prior term.

---

### **References**
- [Original Paper: DPIR - Denoising Prior Driven Image Restoration](https://arxiv.org/abs/1905.04950)
- [DRUNet Official Repository](https://github.com/cszn/DPIR)

---


### **Future Work**
- Add support for other types of degradation.
- Integrate other denoisers into the Plug-and-Play framework.
- Enhance computational efficiency for larger images.
