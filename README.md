# webcam-object-detection

Basic object detection using detr-resnet-101 on the webcam feed.

## Prerequisites

- Python 3.x installed on your system
- `virtualenv` package installed (if not, you can install it using `pip install virtualenv`)

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/stephenswetonic/webcam-object-detection
   ```
2. Navigate to the project directory:

    ```bash
    cd webcam-object-detection
    ```
3. Create a virtual environment (optional but recommended):

    ```bash
    virtualenv venv
    ```
4. Activate the virtual environment:

    ```bash
    source .venv/bin/activate
    ```
5. Install dependencies:

    ```bash
    pip install opencv-python transformers timm pillow
    pip install 'transformers[torch]'
    ```

## Usage

```bash
python video-object-detector.py
```
