import os

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from options.base_options import BaseOptions
from PIL import Image
from scipy.io import savemat
from util.load_mats import load_lm3d
from util.preprocess import align_img

from models import create_model

app = FastAPI(title="Deep3DFaceRecon API", description="Generate 3D face meshes from images")


class APIOptions(BaseOptions):
    """This class includes API options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument("--phase", type=str, default="test", help="train, val, test, etc")
        parser.add_argument(
            "--dataset_mode", type=str, default=None, help="chooses how datasets are loaded. [None | flist]"
        )
        parser.add_argument("--img_folder", required=True, help="Path to folder containing images")
        parser.add_argument("--mesh_folder", required=True, help="Path to folder containing meshes")

        # Dropout and Batchnorm has different behavior during training and test.
        self.isTrain = False
        return parser


# Global variables to store model and configuration
model = None
opt = None
lm3d_std = None
device = None


def initialize_model(rank: int = 0):
    global opt, model, lm3d_std, device

    opt = APIOptions().parse()

    lm3d_std = load_lm3d(opt.bfm_folder)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model = create_model(opt)
    model.setup(opt)
    model.device = device
    model.parallelize()
    model.eval()


def read_data(im_path, lm_path, lm3d_std, to_tensor=True):
    im = Image.open(im_path).convert("RGB")
    W, H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    _, im, lm, _ = align_img(im, lm, lm3d_std)
    if to_tensor:
        im = torch.tensor(np.array(im) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
    return im, lm


def find_landmark_path(img_path):
    """Find corresponding landmark file for an image"""
    global opt
    img_name = os.path.basename(img_path)
    landmark_name = img_name.replace(".png", ".txt").replace(".jpg", ".txt")
    landmark_path = os.path.join(opt.img_folder, "detections", landmark_name)
    return landmark_path


@app.on_event("startup")
async def startup_event():
    """Initialize model when the application starts"""
    initialize_model()


@app.post("/generate-mesh")
async def generate_mesh(filename: str):
    """
    Generate 3D mesh from an image

    Args:
        filename: Name of the image file (e.g., "face.jpg")
    """
    global opt, device
    try:
        # Construct full paths
        img_path = os.path.join(opt.img_folder, filename)
        landmark_path = find_landmark_path(img_path)

        # Check if files exist
        if not os.path.exists(img_path):
            raise HTTPException(status_code=404, detail=f"Image file not found: {img_path}")

        if not os.path.exists(landmark_path):
            raise HTTPException(status_code=404, detail=f"Landmark file not found: {landmark_path}")

        # Generate base name for output files
        base_name = os.path.splitext(filename)[0]

        # Create mesh folder if it doesn't exist
        os.makedirs(opt.mesh_folder, exist_ok=True)

        # Read and preprocess data
        im_tensor, lm_tensor = read_data(img_path, landmark_path, lm3d_std)

        # Prepare data for model
        data = {"imgs": im_tensor.to(device), "lms": lm_tensor.to(device)}

        # Run inference
        model.set_input(data)
        model.test()

        # Save results
        mesh_path = os.path.join(opt.mesh_folder, f"{base_name}.obj")

        model.save_mesh(mesh_path)

        return {
            "status": "success",
            "message": "Mesh generated successfully",
            "mesh_file": mesh_path,
            "input_image": filename,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating mesh: {str(e)}")


print("Running API")
uvicorn.run(app, host="0.0.0.0", port=8000)
