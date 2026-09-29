import glob
from cahal import CAHAL



if __name__ == "__main__":
    # Example usage
    input_file = "./copias.nii.gz"
    for nii in glob.glob("./case/sub*.nii.gz"):
        print(f"Processing file: {nii}")
        input_file = nii

        output_file = CAHAL(input_file, denoise=True, resample=True, output_dir="./output", gpu_memory_limit_gb = 32)
        print(f"Processed file saved at: {output_file}")