
import cv2                    # OpenCV — the industry standard for CV tasks
import numpy as np            # You already know this!
from PIL import Image         # Pillow — simpler image ops, great for format handling
import io


class ImagePreprocessor:
    def __init__(self, target_dpi: int = 300):
        
        self.target_dpi = target_dpi

    def process(self, image_input) -> np.ndarray:
        
        img = self._load_image(image_input)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = self._upscale_if_needed(gray)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        _, binary = cv2.threshold(
            denoised, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        deskewed = self._deskew(binary)

        return deskewed

    def process_to_pil(self, image_input) -> Image.Image:
        processed = self.process(image_input)
        return Image.fromarray(processed)

    def _load_image(self, image_input) -> np.ndarray:
        
        if isinstance(image_input, np.ndarray):
            return image_input                          # Already a numpy array, great

        elif isinstance(image_input, str):
            img = cv2.imread(image_input)               # Load from file path
            if img is None:
                raise FileNotFoundError(f"Could not load image: {image_input}")
            return img

        elif isinstance(image_input, bytes):
            arr = np.frombuffer(image_input, np.uint8)  # Convert bytes to array
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)  # Decode to image

        elif isinstance(image_input, Image.Image):
            return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)

        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

    def _upscale_if_needed(self, gray: np.ndarray) -> np.ndarray:
       
        h, w = gray.shape[:2]
        if w < 1000:
            scale = 1000 / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return gray

    def _deskew(self, binary: np.ndarray) -> np.ndarray:
        
        coords = np.column_stack(np.where(binary > 0))

        if len(coords) < 5:
            return binary   # Not enough pixels to calculate angle, skip

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = 90 + angle

        if abs(angle) > 0.5:
            h, w = binary.shape[:2]
            center = (w // 2, h // 2)

            M = cv2.getRotationMatrix2D(center, angle, scale=1.0)

            binary = cv2.warpAffine(
                binary, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE  # Fill edges with nearby pixels
            )

        return binary

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocessor.py path/to/image.jpg")
        print("\nTesting with a synthetic image instead...")

        test_img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White image
        cv2.putText(test_img, "Hello, Deutschland!", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

        preprocessor = ImagePreprocessor()
        result = preprocessor.process(test_img)
        print(f"Input shape:  {test_img.shape}  (height, width, channels)")
        print(f"Output shape: {result.shape}   (grayscale = 1 channel)")
        print("Preprocessing successful!")
    else:
        preprocessor = ImagePreprocessor()
        result = preprocessor.process(sys.argv[1])
        cv2.imwrite("preprocessed_output.png", result)
        print(f"Saved to preprocessed_output.png  (shape: {result.shape})")
