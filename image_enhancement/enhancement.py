def contrast_enhancement(image, alpha=1.5, beta=0, display_images=True, save=False, output_path=None):
    """
    Applies contrast enhancement to an image.

    Parameters:
        image (numpy.ndarray): The input image.
        alpha (float): Contrast control (1.0 = no change, >1.0 = increase contrast).
        beta (float): Brightness control (0 = no change).
        display_images (bool): Whether to display the original and contrast-enhanced images.
        save (bool): Whether to save the plot as an image.
        output_path (str): Path to save the plot if save is True.

    Returns:
        numpy.ndarray: The contrast-enhanced image.
    """
    contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    if display_images:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Contrast Enhanced Image')
        ax[1].axis('off')
        
        if save and output_path:
            save_plot(fig, output_path)
        else:
            plt.show()
    
    return contrast_image

def color_balance(image, r_gain=1.0, g_gain=1.0, b_gain=1.0, display_images=True, save=False, output_path=None):
    """
    Applies color balance adjustment to an image.

    Parameters:
        image (numpy.ndarray): The input image.
        r_gain (float): Red channel gain.
        g_gain (float): Green channel gain.
        b_gain (float): Blue channel gain.
        display_images (bool): Whether to display the original and color-balanced images.
        save (bool): Whether to save the plot as an image.
        output_path (str): Path to save the plot if save is True.

    Returns:
        numpy.ndarray: The color-balanced image.
    """
    b, g, r = cv2.split(image)
    r = cv2.multiply(r, r_gain)
    g = cv2.multiply(g, g_gain)
    b = cv2.multiply(b, b_gain)
    balanced_image = cv2.merge([b, g, r])
    
    if display_images:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(cv2.cvtColor(balanced_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Color Balanced Image')
        ax[1].axis('off')
        
        if save and output_path:
            save_plot(fig, output_path)
        else:
            plt.show()
    
    return balanced_image

def sharpen_image(image_path, output_path, kernel=None, preprocess_blur=False, output_format='png', display_images=True, comparison=False, save=False):
    """
    Applies sharpening to an image and saves the result. Can handle comparison with different kernels.

    Parameters:
        image_path (str): Path to the input image.
        output_path (str): Path where the output image will be saved. If comparison is True, this is the directory.
        kernel (list of lists or None): Custom sharpening kernel. If None, a default kernel is used.
        preprocess_blur (bool): Whether to apply a slight blur to the image before sharpening.
        output_format (str): Format of the output image file (e.g., 'png', 'jpg'). Default is 'png'.
        display_images (bool): Whether to display the original and sharpened images. Default is True.
        comparison (bool): Whether to iterate over multiple kernels for comparison. Default is False.
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
    
    # Apply preprocessing blur if needed
    if preprocess_blur:
        image_rgb = cv2.GaussianBlur(image_rgb, (3, 3), 0)
    
    # Default sharpening kernel if none provided
    default_kernel = np.array([[0, -0.1, 0],
                               [-0.1, 1.8, -0.1],
                               [0, -0.1, 0]])
    
    if comparison:
        # Ensure output_path is a directory
        os.makedirs(output_path, exist_ok=True)
        
        kernels = kernel if isinstance(kernel, list) else [kernel or default_kernel]
        
        for idx, k in enumerate(kernels):
            # Apply sharpening filter using convolution
            sharpened_image = cv2.filter2D(image_rgb, -1, k)
            
            # Clip pixel values to be within valid range
            sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
            
            # Save the sharpened image
            output_file_path = os.path.join(output_path, f'sharpened_image_kernel{idx}.{output_format}')
            if not cv2.imwrite(output_file_path, cv2.cvtColor(sharpened_image, cv2.COLOR_RGB2BGR)):
                raise IOError(f"Failed to save image to path: {output_file_path}")
            
            if display_images:
                # Plot the original and sharpened images
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                
                # Original Image
                ax[0].imshow(image_rgb)
                ax[0].set_title('Original Image')
                ax[0].axis('off')
                
                # Sharpened Image
                ax[1].imshow(sharpened_image)
                ax[1].set_title(f'Sharpened Image\nKernel Index: {idx}')
                ax[1].axis('off')
                
                if save:
                    plot_filename = os.path.join(output_path, f'comparison_kernel{idx}_plot.{output_format}')
                    save_plot(fig, plot_filename, output_format)
                    print(f"Comparison plot saved as {plot_filename}")
                else:
                    plt.show()
            
            if save:
                print(f"Sharpened image saved as {output_file_path}")
    
    else:
        # Apply sharpening filter using convolution
        sharpened_image = cv2.filter2D(image_rgb, -1, kernel or default_kernel)
        
        # Clip pixel values to be within valid range
        sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
        
        # Save the sharpened image
        output_path_with_format = f"{os.path.splitext(output_path)[0]}.{output_format}"
        if not cv2.imwrite(output_path_with_format, cv2.cvtColor(sharpened_image, cv2.COLOR_RGB2BGR)):
            raise IOError(f"Failed to save image to path: {output_path_with_format}")
        
        if display_images:
            # Plot the original and sharpened images
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            
            # Original Image
            ax[0].imshow(image_rgb)
            ax[0].set_title('Original Image')
            ax[0].axis('off')
            
            # Sharpened Image
            ax[1].imshow(sharpened_image)
            ax[1].set_title(f'Sharpened Image')
            ax[1].axis('off')
            
            if save:
                plot_filename = f"{os.path.splitext(output_path)[0]}_plot.{output_format}"
                save_plot(fig, plot_filename, output_format)
                print(f"Plot saved as {plot_filename}")
            else:
                plt.show()
        
        if save:
            print(f"Sharpened image saved as {output_path_with_format}")

def histogram_equalization(image, display_images=True, save=False, output_path=None):
    """
    Applies histogram equalization to an image to enhance contrast.

    Parameters:
        image (numpy.ndarray): The input image.
        display_images (bool): Whether to display the original and histogram-equalized images.
        save (bool): Whether to save the plot as an image.
        output_path (str): Path to save the plot if save is True.

    Returns:
        numpy.ndarray: The image with histogram equalization applied.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    equalized_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    
    if display_images:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(cv2.cvtColor(equalized_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title('Histogram Equalized Image')
        ax[1].axis('off')
        
        if save and output_path:
            save_plot(fig, output_path)
        else:
            plt.show()
    
    return equalized_image

def histogram_stretching(image, display_images=True, save=False, output_path=None):
    """
    Applies histogram stretching to enhance contrast in an image.

    Parameters:
        image (numpy.ndarray): The input image.
        display_images (bool): Whether to display the original and stretched images.
        save (bool): Whether to save the plot as an image.
        output_path (str): Path to save the plot if save is True.

    Returns:
        numpy.ndarray: The histogram-stretched image.
    """
    # Check if image is grayscale or color
    if len(image.shape) == 3:  # Color image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:  # Grayscale image
        gray = image
    else:
        raise ValueError("Invalid image format. Image must be 2D (grayscale) or 3D (color).")

    min_val, max_val = np.min(gray), np.max(gray)
    stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    if display_images:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(gray, cmap='gray')
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(stretched, cmap='gray')
        ax[1].set_title('Histogram Stretched Image')
        ax[1].axis('off')
        
        if save and output_path:
            save_plot(fig, output_path)
        else:
            plt.show()
    
    return stretched

def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8), display_images=True, save=False, output_path=None):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to an image.

    Parameters:
        image (numpy.ndarray): The input image.
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (tuple): Size of the grid for CLAHE.
        display_images (bool): Whether to display the original and CLAHE-applied images.
        save (bool): Whether to save the plot as an image.
        output_path (str): Path to save the plot if save is True.

    Returns:
        numpy.ndarray: The image with CLAHE applied.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(gray)
    clahe_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    
    if display_images:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax[0].set_title('Original Image')
        ax[0].axis('off')
        
        ax[1].imshow(cv2.cvtColor(clahe_image, cv2.COLOR_BGR2RGB))
        ax[1].set_title('CLAHE Image')
        ax[1].axis('off')
        
        if save and output_path:
            save_plot(fig, output_path)
        else:
            plt.show()
    
    return clahe_image