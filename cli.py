import argparse
import cv2
from odin.image_enhancement.contrast import contrast_enhancement_func
from odin.image_enhancement.color_balance import color_balance_func
from odin.image_enhancement.histogram import histogram_stretching
from odin.image_enhancement.sharpen import sharpen_image
from odin.image_filtering.blur import gaussian_blur
from odin.image_filtering.denoising import denoising
from odin.image_filtering.edge_detection import edge_detection_func
from odin.image_filtering.morphological_operations import amorphological_operations_func
from odin.image_filtering.rescaling import rescaling_func

def main():
    parser = argparse.ArgumentParser(description='Odin Astrophotography Toolbox CLI')

    subparsers = parser.add_subparsers(dest='command')

    # Subparser for contrast enhancement
    contrast_parser = subparsers.add_parser('contrast', help='Apply contrast enhancement')
    contrast_parser.add_argument('image_path', type=str, help='Path to the input image')
    contrast_parser.add_argument('output_path', type=str, help='Path to save the enhanced image')
    contrast_parser.add_argument('--alpha', type=float, default=1.5, help='Contrast control (default: 1.5)')
    contrast_parser.add_argument('--beta', type=float, default=0, help='Brightness control (default: 0)')

    # Subparser for color balance
    color_balance_parser = subparsers.add_parser('color_balance', help='Apply color balance')
    color_balance_parser.add_argument('image_path', type=str, help='Path to the input image')
    color_balance_parser.add_argument('output_path', type=str, help='Path to save the balanced image')
    color_balance_parser.add_argument('--r_gain', type=float, default=1.0, help='Red channel gain (default: 1.0)')
    color_balance_parser.add_argument('--g_gain', type=float, default=1.0, help='Green channel gain (default: 1.0)')
    color_balance_parser.add_argument('--b_gain', type=float, default=1.0, help='Blue channel gain (default: 1.0)')

    # Subparser for histogram stretching
    histogram_parser = subparsers.add_parser('histogram_stretching', help='Apply histogram stretching')
    histogram_parser.add_argument('image_path', type=str, help='Path to the input image')
    histogram_parser.add_argument('output_path', type=str, help='Path to save the stretched image')

    # Subparser for sharpening
    sharpen_parser = subparsers.add_parser('sharpen', help='Apply sharpening')
    sharpen_parser.add_argument('image_path', type=str, help='Path to the input image')
    sharpen_parser.add_argument('output_path', type=str, help='Path to save the sharpened image')

    # Subparser for Gaussian blur
    blur_parser = subparsers.add_parser('blur', help='Apply Gaussian blur')
    blur_parser.add_argument('image_path', type=str, help='Path to the input image')
    blur_parser.add_argument('output_path', type=str, help='Path to save the blurred image')
    blur_parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size (default: 5)')
    blur_parser.add_argument('--sigma', type=float, default=1.0, help='Sigma value (default: 1.0)')

    # Subparser for denoising
    denoising_parser = subparsers.add_parser('denoise', help='Apply denoising')
    denoising_parser.add_argument('image_path', type=str, help='Path to the input image')
    denoising_parser.add_argument('output_path', type=str, help='Path to save the denoised image')

    # Subparser for edge detection
    edge_detection_parser = subparsers.add_parser('edge_detection', help='Apply edge detection')
    edge_detection_parser.add_argument('image_path', type=str, help='Path to the input image')
    edge_detection_parser.add_argument('output_path', type=str, help='Path to save the edge-detected image')
    edge_detection_parser.add_argument('--low_threshold', type=int, default=50, help='Low threshold for Canny (default: 50)')
    edge_detection_parser.add_argument('--high_threshold', type=int, default=150, help='High threshold for Canny (default: 150)')

    # Subparser for morphological operations
    morph_parser = subparsers.add_parser('morph', help='Apply morphological operations')
    morph_parser.add_argument('image_path', type=str, help='Path to the input image')
    morph_parser.add_argument('output_path', type=str, help='Path to save the morphologically processed image')
    morph_parser.add_argument('--operation', type=str, choices=['erode', 'dilate', 'open', 'close'], required=True, help='Morphological operation')
    morph_parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size (default: 5)')
    morph_parser.add_argument('--iterations', type=int, default=1, help='Number of iterations (default: 1)')

    # Subparser for rescaling
    rescale_parser = subparsers.add_parser('rescale', help='Apply rescaling')
    rescale_parser.add_argument('image_path', type=str, help='Path to the input image')
    rescale_parser.add_argument('output_path', type=str, help='Path to save the rescaled image')
    rescale_parser.add_argument('--scale_percent', type=int, default=100, help='Scale percentage (default: 100)')

    args = parser.parse_args()

    if args.command == 'contrast':
        image = cv2.imread(args.image_path)
        enhanced_image = apply_contrast_enhancement_func(image, alpha=args.alpha, beta=args.beta)
        cv2.imwrite(args.output_path, enhanced_image)

    elif args.command == 'color_balance':
        image = cv2.imread(args.image_path)
        balanced_image = apply_color_balance_func(image, r_gain=args.r_gain, g_gain=args.g_gain, b_gain=args.b_gain)
        cv2.imwrite(args.output_path, balanced_image)

    elif args.command == 'histogram_stretching':
        image = cv2.imread(args.image_path)
        stretched_image = apply_histogram_stretching(image)
        cv2.imwrite(args.output_path, stretched_image)

    elif args.command == 'sharpen':
        sharpen_image(args.image_path, args.output_path)

    elif args.command == 'blur':
        apply_gaussian_blur(args.image_path, kernel_size=args.kernel_size, sigma=args.sigma, output_path=args.output_path)

    elif args.command == 'denoise':
        image = cv2.imread(args.image_path)
        denoised_image = apply_denoising(image)
        cv2.imwrite(args.output_path, denoised_image)

    elif args.command == 'edge_detection':
        image = cv2.imread(args.image_path)
        edge_detected_image = apply_edge_detection_func(image, low_threshold=args.low_threshold, high_threshold=args.high_threshold)
        cv2.imwrite(args.output_path, edge_detected_image)

    elif args.command == 'morph':
        image = cv2.imread(args.image_path)
        morph_image = apply_morphological_operations_func(image, operation=args.operation, kernel_size=args.kernel_size, iterations=args.iterations)
        cv2.imwrite(args.output_path, morph_image)

    elif args.command == 'rescale':
        image = cv2.imread(args.image_path)
        rescaled_image = apply_rescaling_func(image, scale_percent=args.scale_percent)
        cv2.imwrite(args.output_path, rescaled_image)

if __name__ == '__main__':
    main()
