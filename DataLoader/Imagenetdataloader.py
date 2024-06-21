from PIL import Image
import os
import numpy as np
from tinygrad import Tensor

def load_custom_data(data_dir, image_size=(28, 28)):
    from PIL import Image
    import os
    import numpy as np

    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            with Image.open(img_path) as img:
                img = img.resize(image_size).convert('L')  # Resize and convert to grayscale
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values and ensure float32
                images.append(img_array)
                labels.append(class_to_idx[class_name])
    
    images = np.expand_dims(np.array(images, dtype=np.float32), axis=1) # Add channel dimension and ensure float32
    labels = np.array(labels, dtype=np.int64)  # Ensure labels are int64
    
    return Tensor(images), Tensor(labels)
