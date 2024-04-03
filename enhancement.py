# import os
# import subprocess
# import argparse

# def enhance_images(input_folder="LR/*", output_folder="results/", num_iterations = 3):
#     # Run test.py to generate initial enhanced images
#     subprocess.run(["python", "test_ESRGAN.py", "--input_folder", input_folder, "--output_folder", output_folder])

#     # Perform multiple enhancements
#     for i in range(num_iterations):
#         # Update input and output folder paths
#         input_folder = output_folder
#         output_folder = f"enhanced_{i+1}/"
#         os.makedirs(output_folder, exist_ok=True)

#         # Iterate through images in the input folder
#         for filename in os.listdir(input_folder):
#             if filename.endswith(".png") or filename.endswith(".jpg"):  # Adjust file extensions as needed
#                 input_file = os.path.join(input_folder, filename)
#                 output_file = os.path.join(output_folder, filename)
#                 subprocess.run(["python", "test_ESRGAN.py", "--input", input_file, "--output", output_file])

#     print("Enhancement complete.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Enhance images multiple times.")
#     parser.add_argument("--input_folder", type=str, help="Path to the input folder containing images.")
#     parser.add_argument("--output_folder", type=str, help="Path to the output folder to store enhanced images.")
#     parser.add_argument("--num_iterations", type=int, default=5, help="Number of enhancement iterations.")
#     args = parser.parse_args()

#     enhance_images(args.input_folder, args.output_folder, args.num_iterations)


# import os.path as osp
# import os
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import argparse

# def enhance_images(input_folder="blurry_text_images/*", output_folder="results/", num_iterations=3, model_path='models/RRDB_ESRGAN_x4.pth'):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = arch.RRDBNet(3, 3, 64, 23, gc=32)
#     model.load_state_dict(torch.load(model_path), strict=True)
#     model.eval()
#     model = model.to(device)

#     print('Model path {:s}. \nTesting...'.format(model_path))

#     for iteration in range(num_iterations + 1):  # +1 for initial enhancement
#         idx = 0
#         if iteration == 0:
#             current_input_folder = input_folder
#         else:
#             current_input_folder = os.path.join(output_folder, f"enhanced_{iteration}/")

#         for path in glob.glob(current_input_folder):
#             idx += 1
#             base = osp.splitext(osp.basename(path))[0]
#             print(f"Iteration {iteration}, Processing image {idx}: {base}")

#             # Read images with error handling
#             img = cv2.imread(path, cv2.IMREAD_COLOR)
#             print(img)
#             if img is None:
#                 print(f"Error reading image: {path}")
#                 continue

#             img = img * 1.0 / 255
#             img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#             img_LR = img.unsqueeze(0)
#             img_LR = img_LR.to(device)

#             with torch.no_grad():
#                 output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#             output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#             output = (output * 255.0).round()

#             # Save enhanced image
#             if iteration == num_iterations:
#                 cv2.imwrite(osp.join(output_folder, f"{base}_enhanced_{iteration}.png"), output)
#             else:
#                 os.makedirs(osp.join(output_folder, f"enhanced_{iteration + 1}/"), exist_ok=True)
#                 cv2.imwrite(osp.join(output_folder, f"enhanced_{iteration + 1}/{base}_rlt.png"), output)

#     print("Enhancement complete.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Enhance images multiple times using ESRGAN.")
#     parser.add_argument("--input_folder", type=str, default="blurry_text_images/*", help="Path to the input folder containing images.")
#     parser.add_argument("--output_folder", type=str, default="results/", help="Path to the output folder to store enhanced images.")
#     parser.add_argument("--num_iterations", type=int, default=3, help="Number of enhancement iterations.")
#     parser.add_argument("--model_path", type=str, default="models/RRDB_ESRGAN_x4.pth", help="Path to the ESRGAN model.")
#     args = parser.parse_args()

#     enhance_images(args.input_folder, args.output_folder, args.num_iterations, args.model_path)

# import os
# import subprocess
# import argparse

# def enhance_images(input_folder="LR/*", output_folder="results/", num_iterations = 3):
#     # Run test.py to generate initial enhanced images
#     subprocess.run(["python", "test_ESRGAN.py", "--input_folder", input_folder, "--output_folder", output_folder])

#     # Perform multiple enhancements
#     for i in range(num_iterations):
#         # Update input and output folder paths
#         input_folder = output_folder
#         output_folder = f"enhanced_{i+1}/"
#         os.makedirs(output_folder, exist_ok=True)

#         # Iterate through images in the input folder
#         for filename in os.listdir(input_folder):
#             if filename.endswith(".png") or filename.endswith(".jpg"):  # Adjust file extensions as needed
#                 input_file = os.path.join(input_folder, filename)
#                 output_file = os.path.join(output_folder, filename)
#                 subprocess.run(["python", "test_ESRGAN.py", "--input", input_file, "--output", output_file])

#     print("Enhancement complete.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Enhance images multiple times.")
#     parser.add_argument("--input_folder", type=str, help="Path to the input folder containing images.")
#     parser.add_argument("--output_folder", type=str, help="Path to the output folder to store enhanced images.")
#     parser.add_argument("--num_iterations", type=int, default=5, help="Number of enhancement iterations.")
#     args = parser.parse_args()

#     enhance_images(args.input_folder, args.output_folder, args.num_iterations)


# import os.path as osp
# import os
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import argparse

# def enhance_images(input_folder="blurry_text_images/*", output_folder="results/", num_iterations=3, model_path='models/RRDB_ESRGAN_x4.pth'):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = arch.RRDBNet(3, 3, 64, 23, gc=32)
#     model.load_state_dict(torch.load(model_path), strict=True)
#     model.eval()
#     model = model.to(device)

#     print('Model path {:s}. \nTesting...'.format(model_path))

#     for iteration in range(num_iterations + 1):  # +1 for initial enhancement
#         idx = 0
#         if iteration == 0:
#             current_input_folder = input_folder
#         else:
#             current_input_folder = os.path.join(output_folder, f"enhanced_{iteration}/")

#         for path in glob.glob(current_input_folder):
#             idx += 1
#             base = osp.splitext(osp.basename(path))[0]
#             print(f"Iteration {iteration}, Processing image {idx}: {base}")

#             # Read images with error handling
#             img = cv2.imread(path, cv2.IMREAD_COLOR)
#             print(img)
#             if img is None:
#                 print(f"Error reading image: {path}")
#                 continue

#             img = img * 1.0 / 255
#             img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#             img_LR = img.unsqueeze(0)
#             img_LR = img_LR.to(device)

#             with torch.no_grad():
#                 output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#             output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
#             output = (output * 255.0).round()

#             # Save enhanced image
#             if iteration == num_iterations:
#                 cv2.imwrite(osp.join(output_folder, f"{base}_enhanced_{iteration}.png"), output)
#             else:
#                 os.makedirs(osp.join(output_folder, f"enhanced_{iteration + 1}/"), exist_ok=True)
#                 cv2.imwrite(osp.join(output_folder, f"enhanced_{iteration + 1}/{base}_rlt.png"), output)

#     print("Enhancement complete.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Enhance images multiple times using ESRGAN.")
#     parser.add_argument("--input_folder", type=str, default="blurry_text_images/*", help="Path to the input folder containing images.")
#     parser.add_argument("--output_folder", type=str, default="results/", help="Path to the output folder to store enhanced images.")
#     parser.add_argument("--num_iterations", type=int, default=3, help="Number of enhancement iterations.")
#     parser.add_argument("--model_path", type=str, default="models/RRDB_ESRGAN_x4.pth", help="Path to the ESRGAN model.")
#     args = parser.parse_args()

#     enhance_images(args.input_folder, args.output_folder, args.num_iterations, args.model_path)

# import os
# import os.path as osp
# import glob
# import cv2
# import numpy as np
# import torch
# import RRDBNet_arch as arch
# import argparse

# def enhance_images(input_folder="LR", output_folder="results/", num_iterations=3, model_path='models/RRDB_ESRGAN_x4.pth'):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     model = arch.RRDBNet(3, 3, 64, 23, gc=32)
#     model.load_state_dict(torch.load(model_path), strict=True)
#     model.eval()
#     model = model.to(device)

#     print('Model path {:s}. \nTesting...'.format(model_path))

#     for iteration in range(num_iterations):  # No need to enhance on the last iteration
#         idx = 0
#         if iteration == 0:
#             current_input_folder = input_folder
#         else:
#             current_input_folder = osp.join(output_folder, f"enhanced_{iteration}/")

#         # Filter out directories from glob results
#         input_files = [f for f in glob.glob(current_input_folder+'/*') if osp.isfile(f)]

#         for path in input_files:
#             idx += 1
#             base = osp.splitext(osp.basename(path))[0]
#             print(f"Iteration {iteration + 1}, Processing image {idx}: {base}")

#             # Read images with error handling
#             try:
#                 img = cv2.imread(path, cv2.IMREAD_COLOR)
#                 if img is None:
#                     raise Exception(f"Failed to read image: {path}")
#             except Exception as e:
#                 print(f"Error reading image: {e}")
#                 continue

#             img = img * 1.0 / 255
#             img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
#             img_LR = img.unsqueeze(0)
#             img_LR = img_LR.to(device)

#             with torch.no_grad():
#                 output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
#             output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            
#             # Resize output image to the same size as the input image
#             # target_height, target_width, _ = img.shape
#             # output = cv2.resize(output, (target_width, target_height))
            
#             # Save enhanced image
#             os.makedirs(osp.join(output_folder, f"enhanced_{iteration + 1}/"), exist_ok=True)
#             cv2.imwrite(osp.join(output_folder, f"enhanced_{iteration + 1}/{base}_rlt.png"), (output * 255).astype(np.uint8))

#     print("Enhancement complete.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Enhance images multiple times using ESRGAN.")
#     parser.add_argument("--input_folder", type=str, default="LR", help="Path to the input folder containing images.")
#     parser.add_argument("--output_folder", type=str, default="results/", help="Path to the output folder to store enhanced images.")
#     parser.add_argument("--num_iterations", type=int, help="Number of enhancement iterations.")
#     parser.add_argument("--model_path", type=str, default="models/RRDB_ESRGAN_x4.pth", help="Path to the ESRGAN model.")
#     args = parser.parse_args()

#     enhance_images(args.input_folder, args.output_folder, args.num_iterations, args.model_path)


import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import argparse

def enhance_images(input_folder="LR", output_folder="results/", model_path='models/RRDB_ESRGAN_x4.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print('Model path {:s}. \nTesting...'.format(model_path))

    # Filter out directories from glob results
    input_files = [f for f in glob.glob(input_folder+'/*') if osp.isfile(f)]

    for path in input_files:
        base = osp.splitext(osp.basename(path))[0]
        print(f"Processing image: {base}")

        # Read images with error handling
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise Exception(f"Failed to read image: {path}")
        except Exception as e:
            print(f"Error reading image: {e}")
            continue

        img = img * 1.0 / 255
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        
        # Save enhanced image
        os.makedirs(output_folder, exist_ok=True)
        cv2.imwrite(osp.join(output_folder, f"{base}_rlt.png"), (output * 255).astype(np.uint8))

    print("Enhancement complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhance images using ESRGAN.")
    parser.add_argument("--input_folder", type=str, default="LR", help="Path to the input folder containing images.")
    parser.add_argument("--output_folder", type=str, default="results/", help="Path to the output folder to store enhanced images.")
    parser.add_argument("--model_path", type=str, default="models/RRDB_ESRGAN_x4.pth", help="Path to the ESRGAN model.")
    args = parser.parse_args()

    enhance_images(args.input_folder, args.output_folder, args.model_path)

