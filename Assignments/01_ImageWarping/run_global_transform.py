import gradio as gr
import cv2
import numpy as np

# Function to convert 2x3 affine matrix to 3x3 for matrix multiplication
def to_3x3(affine_matrix):
    return np.vstack([affine_matrix, [0, 0, 1]])


# Function to apply transformations based on user inputs
def apply_transform(
    image, scale, rotation, translation_x, translation_y, flip_horizontal, flip_vertical
):

    # Convert the image from PIL format to a NumPy array
    image = np.array(image)
    # Pad the image to avoid boundary issues
    pad_size = min(image.shape[0], image.shape[1]) // 2
    image_new = np.zeros(
        (pad_size * 2 + image.shape[0], pad_size * 2 + image.shape[1], 3),
        dtype=np.uint8,
    ) + np.array((255, 255, 255), dtype=np.uint8).reshape(1, 1, 3)
    image_new[
        pad_size : pad_size + image.shape[0], pad_size : pad_size + image.shape[1]
    ] = image
    image = np.array(image_new)
    transformed_image = np.array(image)

    ### FILL: Apply Composition Transform
    # Note: for scale and rotation, implement them around the center of the image (围绕图像中心进行放缩和旋转)

    # Get the center of the image
    image_width = image.shape[1]
    image_height = image.shape[0]
    center = (image_width // 2, image_height // 2)

    # 1. Scale
    scale_matrix = cv2.getRotationMatrix2D(center, 0, scale)  # dimension: 2x3
    transformed_image = cv2.warpAffine(image, scale_matrix, (image_width, image_height))

    # 2. Rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1)
    transformed_image = cv2.warpAffine(
        transformed_image, rotation_matrix, (image_width, image_height)
    )

    # 3. Translation
    translation_matrix = np.float32(
        [[1, 0, translation_x], [0, 1, -translation_y]]
    )  # remark: -translation_y
    transformed_image = cv2.warpAffine(
        transformed_image, translation_matrix, (image_width, image_height)
    )

    # 4. Flip
    if flip_horizontal:
        transformed_image = cv2.flip(transformed_image, 1)
    
    if flip_vertical:
        transformed_image = cv2.flip(transformed_image, 0)

    return transformed_image


# Gradio Interface
def interactive_transform():
    with gr.Blocks(gr.themes.Base()) as demo:
        gr.Markdown(
            "<h1 style='text-align: center'>Image Transformation Playground</h1>"
        )

        # Define the layout
        with gr.Row():
            # Left: Image input and sliders
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")

                scale = gr.Slider(
                    minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Scale"
                )
                rotation = gr.Slider(
                    minimum=-180,
                    maximum=180,
                    step=1,
                    value=0,
                    label="Rotation (degrees)",
                )
                translation_x = gr.Slider(
                    minimum=-300, maximum=300, step=10, value=0, label="Translation X"
                )
                translation_y = gr.Slider(
                    minimum=-300, maximum=300, step=10, value=0, label="Translation Y"
                )
                with gr.Row():
                    flip_horizontal = gr.Checkbox(label="Flip Horizontal")
                    flip_vertical = gr.Checkbox(label="Flip Vertical")

            # Right: Output image
            image_output = gr.Image(label="Transformed Image")

        # Automatically update the output when any slider or checkbox is changed
        inputs = [
            image_input,
            scale,
            rotation,
            translation_x,
            translation_y,
            flip_horizontal,
            flip_vertical,
        ]

        # Link inputs to the transformation function
        image_input.change(apply_transform, inputs, image_output)
        scale.change(apply_transform, inputs, image_output)
        rotation.change(apply_transform, inputs, image_output)
        translation_x.change(apply_transform, inputs, image_output)
        translation_y.change(apply_transform, inputs, image_output)
        flip_horizontal.change(apply_transform, inputs, image_output)
        flip_vertical.change(apply_transform, inputs, image_output)

    return demo


demo = interactive_transform()

if __name__ == "__main__":
    # Launch the Gradio interface
    demo.launch(show_error=True)
