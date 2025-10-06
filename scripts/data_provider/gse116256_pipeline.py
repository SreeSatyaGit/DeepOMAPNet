"""
run_gse116256_processing.py - Run GSE116256 dataset processing

This script processes the GSE116256 dataset and prepares it for testing with the trained models.
"""

import os
import sys
import subprocess

def main():
    """
    Main function to run GSE116256 processing
    """
    print("="*60)
    print("GSE116256 Dataset Processing")
    print("="*60)

    data_dir = "/projects/vanaja_lab/satya/Datasets/GSE116256"
    if not os.path.exists(data_dir):
        print(f"❌ Error: Data directory {data_dir} does not exist!")
        print("Please ensure the GSE116256 dataset is available at the specified path.")
        return

    if not os.path.exists("load_gse116256.py"):
        print("❌ Error: load_gse116256.py not found!")
        print("Please run this script from the DeepOMAPNet directory.")
        return

    print(f"Data directory: {data_dir}")
    print("Processing dataset...")

    try:
        result = subprocess.run([
            sys.executable, "load_gse116256.py"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Dataset processing completed successfully!")
            print("\nOutput:")
            print(result.stdout)

            output_file = "GSE116256_combined.h5ad"
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file) / 1024**2
                print(f"\n📁 Output file created: {output_file}")
                print(f"   File size: {file_size:.1f} MB")

                print(f"\n🚀 Next steps:")
                print(f"   1. Open Test_New_Dataset.ipynb")
                print(f"   2. Uncomment the GSE116256 loading section in cell 2")
                print(f"   3. Run the notebook to test the trained models on this dataset")
            else:
                print("❌ Warning: Output file not found after processing")
        else:
            print("❌ Error during processing:")
            print(result.stderr)

    except Exception as e:
        print(f"❌ Error running processing script: {e}")

if __name__ == "__main__":
    main()
