# <img src="https://github.com/ZachPereira14/odin/blob/main/odin_icon.ico?raw=true" alt="Odin Icon" width="50" height="50"> <span style="font-size: 32px;">Odin -- Astrophotography Processing Toolbox</span>



Odin is a Python package designed for advanced image processing, specifically tailored for astrophotography. It provides a collection of tools to enhance, filter, and analyze astronomical images.

## Features

- **Image Enhancement:** Contrast enhancement, color balance, histogram stretching, sharpening.
- **Image Filtering:** Gaussian blur, denoising, edge detection, morphological operations, rescaling.
- **Command-Line Interface (CLI):** Execute various image processing tasks directly from the command line.

## Installation

To install the `odin` package, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/odin.git
   cd odin
   ```

2. **Install Dependencies:**

   Make sure you have Python 3.7 or higher installed. You can install the required dependencies using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Install the Package:**

   Install the package locally using:

   ```bash
   pip install .
   ```

## Usage

### CLI Commands

The `odin-cli` provides several commands for different image processing tasks. Here’s how you can use them:

- **Contrast Enhancement**

  ```bash
  odin-cli contrast <image_path> <output_path> [--alpha ALPHA] [--beta BETA]
  ```

  - `image_path`: Path to the input image.
  - `output_path`: Path to save the enhanced image.
  - `--alpha`: Contrast control (default: 1.5).
  - `--beta`: Brightness control (default: 0).

- **Color Balance**

  ```bash
  odin-cli color_balance <image_path> <output_path> [--r_gain R_GAIN] [--g_gain G_GAIN] [--b_gain B_GAIN]
  ```

  - `image_path`: Path to the input image.
  - `output_path`: Path to save the balanced image.
  - `--r_gain`: Red channel gain (default: 1.0).
  - `--g_gain`: Green channel gain (default: 1.0).
  - `--b_gain`: Blue channel gain (default: 1.0).

- **Histogram Stretching**

  ```bash
  odin-cli histogram_stretching <image_path> <output_path>
  ```

  - `image_path`: Path to the input image.
  - `output_path`: Path to save the stretched image.

- **Sharpening**

  ```bash
  odin-cli sharpen <image_path> <output_path>
  ```

  - `image_path`: Path to the input image.
  - `output_path`: Path to save the sharpened image.

- **Gaussian Blur**

  ```bash
  odin-cli blur <image_path> <output_path> [--kernel_size KERNEL_SIZE] [--sigma SIGMA]
  ```

  - `image_path`: Path to the input image.
  - `output_path`: Path to save the blurred image.
  - `--kernel_size`: Kernel size (default: 5).
  - `--sigma`: Sigma value (default: 1.0).

- **Denoising**

  ```bash
  odin-cli denoise <image_path> <output_path>
  ```

  - `image_path`: Path to the input image.
  - `output_path`: Path to save the denoised image.

- **Edge Detection**

  ```bash
  odin-cli edge_detection <image_path> <output_path> [--low_threshold LOW_THRESHOLD] [--high_threshold HIGH_THRESHOLD]
  ```

  - `image_path`: Path to the input image.
  - `output_path`: Path to save the edge-detected image.
  - `--low_threshold`: Low threshold for Canny (default: 50).
  - `--high_threshold`: High threshold for Canny (default: 150).

- **Morphological Operations**

  ```bash
  odin-cli morph <image_path> <output_path> --operation OPERATION [--kernel_size KERNEL_SIZE] [--iterations ITERATIONS]
  ```

  - `image_path`: Path to the input image.
  - `output_path`: Path to save the processed image.
  - `--operation`: Morphological operation (`erode`, `dilate`, `open`, `close`).
  - `--kernel_size`: Kernel size (default: 5).
  - `--iterations`: Number of iterations (default: 1).

- **Rescaling**

  ```bash
  odin-cli rescale <image_path> <output_path> [--scale_percent SCALE_PERCENT]
  ```

  - `image_path`: Path to the input image.
  - `output_path`: Path to save the rescaled image.
  - `--scale_percent`: Scale percentage (default: 100).

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the Repository** and create your branch from `main`.
2. **Make Changes** and test thoroughly.
3. **Create a Pull Request** with a clear description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please contact [zacharypereira14@gmail.com](mailto:zacharypereira14@gmail.com).
