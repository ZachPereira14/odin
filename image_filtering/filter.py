def gaussian_blur(image_path, kernel_size, sigma, output_path, output_format='png', display_images=True, comparison=False, save=False):
    """
    Applies Gaussian blur to an image and saves the result. Can process a single set of parameters or iterate over lists if comparison is True.
    Optionally save the plotted images.

    Parameters:
        image_path (str): Path to the input image.
        kernel_size (int or list): Size of the Gaussian kernel (must be odd) or a list of kernel sizes.
        sigma (float or list): Standard deviation of the Gaussian kernel or a list of sigma values.
        output_path (str): Path where the output image will be saved. If comparison is True, this is the directory.
        output_format (str): Format of the output image file (e.g., 'png', 'jpg'). Default is 'png'.
        display_images (bool): Whether to display the original and blurred images. Default is True.
        comparison (bool): Whether to iterate over lists of kernel sizes and sigma values. Default is False.
        save (bool): Whether to save the plots as images. Default is False.
    """
    # Check if the image file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at the path: {image_path}")
    
    # Load the image
    image = cv2.imread(image_path)
    
    # Check if the image was loaded successfully
    if image is None:
        raise ValueError(f"Failed to load image. Check the file format or path: {image_path}")
    
    # Convert from BGR to RGB (OpenCV loads images in BGR format by default)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if comparison:
        # Ensure kernel_size and sigma are lists
        if not isinstance(kernel_size, list):
            kernel_size = [kernel_size]
        if not isinstance(sigma, list):
            sigma = [sigma]
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Iterate over kernel sizes and sigma values
        for k in kernel_size:
            for s in sigma:
                # Apply Gaussian blur
                blurred_image = cv2.GaussianBlur(image_rgb, (k, k), s)
                
                # Save the blurred image
                output_file_path = os.path.join(output_path, f'blurred_image_kernel{k}_sigma{s}.{output_format}')
                cv2.imwrite(output_file_path, cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))
                
                if display_images:
                    # Plot the original and blurred images
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                    
                    # Original Image
                    ax[0].imshow(image_rgb)
                    ax[0].set_title(f'Original Image\nSize: {image_rgb.shape[1]}x{image_rgb.shape[0]}')
                    ax[0].axis('off')
                    
                    # Blurred Image
                    ax[1].imshow(blurred_image)
                    ax[1].set_title(f'Blurred Image\nKernel Size: {k}, Sigma: {s}')
                    ax[1].axis('off')
                    
                    if save:
                        plot_filename = os.path.join(output_path, f'comparison_kernel{k}_sigma{s}_plot.{output_format}')
                        save_plot(fig, plot_filename, output_format)
                        print(f"Comparison plot saved as {plot_filename}")
                    else:
                        plt.show()
                    
                if save:
                    print(f"Blurred image saved as {output_file_path}")
    
    else:
        # Apply Gaussian Blur for single set of parameters
        blurred_image = cv2.GaussianBlur(image_rgb, (kernel_size, kernel_size), sigma)
        
        # Save the blurred image
        output_path_with_format = f"{os.path.splitext(output_path)[0]}.{output_format}"
        cv2.imwrite(output_path_with_format, cv2.cvtColor(blurred_image, cv2.COLOR_RGB2BGR))
        
        if display_images:
            # Plot the original and blurred images
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original Image
            ax[0].imshow(image_rgb)
            ax[0].set_title(f'Original Image\nSize: {image_rgb.shape[1]}x{image_rgb.shape[0]}')
            ax[0].axis('off')
            
            # Blurred Image
            ax[1].imshow(blurred_image)
            ax[1].set_title(f'Blurred Image\nKernel Size: {kernel_size}, Sigma: {sigma}')
            ax[1].axis('off')
            
            if save:
                plot_filename = f"{os.path.splitext(output_path)[0]}_plot.{output_format}"
                save_plot(fig, plot_filename, output_format)
                print(f"Plot saved as {plot_filename}")
            else:
                plt.show()
        
        if save:
            print(f"Blurred image saved as {output_path_with_format}")
            
def denoising(image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21, display_images=True, save=False, output_path=None):
    """
    Applies denoising to an image using Non-Local Means Denoising.

    Parameters:
        image (numpy.ndarray): The input image.
        h (float): Filter strength for luminance component.
        hColor (float): Filter strength for color components.
        templateWindowSize (int): Size of the template patch used for denoising.
        searchWindowSize (int): Size of the window used for searching similar patches.
        display_images (bool): Whether to display the original and denoised images.
        save (bool): Whether to save the plot as an image.
        output_path (str): Path to save the plot if save is True.

    Returns:
        numpy.ndarray: The denoised image.
    """
    # Check if image is grayscale or color
    if len(image.shape) == 3:  # Color image
        denoised_image = cv2.fastNlMeansDenoisingColored(image, None, hColor, hColor, templateWindowSize, searchWindowSize)
    elif len(image.shape) == 2:  # Grayscale image
        denoised_image = cv2.fastNlMeansDenoising(image, None, h, templateWindowSize, searchWindowSize)
    else:
        raise ValueError("Invalid image format. Image must be 2D (grayscale) or 3D (color).")

    if display_images:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image, cmap='gray' if len(image.shape) == 2 else None)
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB) if len(denoised_image.shape) == 3 else denoised_image, cmap='gray' if len(denoised_image.shape) == 2 else None)
        ax[1].set_title('Denoised Image')
        ax[1].axis('off')
        
        if save and output_path:
            save_plot(fig, output_path)
        else:
            plt.show()
    
    return denoised_image

def edge_detection(image, low_threshold=50, high_threshold=150, display_images=True, save=False, output_path=None):
    """
    Applies edge detection to an image.

    Parameters:
        image (numpy.ndarray): The input image.
        low_threshold (int): Low threshold for Canny edge detection.
        high_threshold (int): High threshold for Canny edge detection.
        display_images (bool): Whether to display the original and edge-detected images.
        save (bool): Whether to save the plot as an image.
        output_path (str): Path to save the plot if save is True.

    Returns:
        numpy.ndarray: The image with edges detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    if display_images:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(cv2.cvtColor(edges_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Edge Detection')
        ax[1].axis('off')
        
        if save and output_path:
            save_plot(fig, output_path)
        else:
            plt.show()
    
    return edges_image

def morphological_operations(image, operation='erode', kernel_size=5, iterations=1, display_images=True, save=False, output_path=None):
    """
    Applies morphological operations to an image.

    Parameters:
        image (numpy.ndarray): The input image.
        operation (str): The morphological operation ('erode', 'dilate', 'open', 'close').
        kernel_size (int): Size of the kernel used for the operation.
        iterations (int): Number of iterations for the operation.
        display_images (bool): Whether to display the original and processed images.
        save (bool): Whether to save the plot as an image.
        output_path (str): Path to save the plot if save is True.

    Returns:
        numpy.ndarray: The image after morphological operations.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'erode':
        morph_image = cv2.erode(gray, kernel, iterations=iterations)
    elif operation == 'dilate':
        morph_image = cv2.dilate(gray, kernel, iterations=iterations)
    elif operation == 'open':
        morph_image = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'close':
        morph_image = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    else:
        raise ValueError("Unsupported morphological operation. Choose from 'erode', 'dilate', 'open', 'close'.")
    
    morph_image = cv2.cvtColor(morph_image, cv2.COLOR_GRAY2BGR)
    
    if display_images:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(cv2.cvtColor(morph_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title(f'Morphological Operation: {operation}')
        ax[1].axis('off')
        
        if save and output_path:
            save_plot(fig, output_path)
        else:
            plt.show()
    
    return morph_image

def rescaling(image, scale_percent=100, display_images=True, save=False, output_path=None):
    """
    Rescales an image to a specified percentage of its original size.

    Parameters:
        image (numpy.ndarray): The input image.
        scale_percent (float): The percentage by which to scale the image.
        display_images (bool): Whether to display the original and rescaled images.
        save (bool): Whether to save the plot as an image.
        output_path (str): Path to save the plot if save is True.

    Returns:
        numpy.ndarray: The rescaled image.
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    rescaled_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
    
    if display_images:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(cv2.cvtColor(rescaled_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title(f'Rescaled Image ({scale_percent}%)')
        ax[1].axis('off')
        
        if save and output_path:
            save_plot(fig, output_path)
        else:
            plt.show()
    
    return rescaled_image