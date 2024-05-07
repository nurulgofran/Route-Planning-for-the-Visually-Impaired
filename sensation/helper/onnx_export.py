import torch
import argparse
import torch.onnx
from segmentation import DeepLabV3PResNet34

# We use 8 classes of cityscapes
num_classes = 8


def export_to_onnx(pytorch_model_path, onnx_model_path):
    # Load the PyTorch model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(pytorch_model_path, map_location=device)
    model = DeepLabV3PResNet34()
    
    model.load_state_dict(state_dict)

    model.eval()

    # Adjust the dummy input to match the model's expected input size
    dummy_input = torch.randn(
        1, 3, 256, 512
    )  # Adjusted for an image of height 256 and width 512

    # Export the model
    torch.onnx.export(
        model,  # model being run
        dummy_input,  # model input (or a tuple for multiple inputs)
        onnx_model_path,  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX format")
    parser.add_argument(
        "--pytorch",
        type=str,
        default="model.pth",
        help="Path to the PyTorch model file",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="model.onnx",
        help="Path where the ONNX model will be stored",
    )

    args = parser.parse_args()

    export_to_onnx(args.pytorch, args.onnx)
