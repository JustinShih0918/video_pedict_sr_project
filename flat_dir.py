import os
import shutil
import sys

def flat_with_leaf(rootFolder, outputFolder):
    os.makedirs(outputFolder, exist_ok=True)
    
    leaf_folders = []
    
    for dirpath, dirnames, filenames in os.walk(rootFolder):
        png_files = [f for f in filenames if f.endswith('.png')]
        
        if png_files:
            is_leaf = True
            for dirname in dirnames:
                sub_path = os.path.join(dirpath, dirname)
                for _, _, sub_files in os.walk(sub_path):
                    if any(f.endswith('.png') for f in sub_files):
                        is_leaf = False
                        break
                if not is_leaf:
                    break
            
            if is_leaf:
                leaf_folders.append((dirpath, png_files))
    
    for index, (folder_path, png_files) in enumerate(leaf_folders, 1):
        folder_name = f"{index:04d}"
        output_path = os.path.join(outputFolder, folder_name)
        os.makedirs(output_path, exist_ok=True)
        
        for png_file in png_files:
            src_file = os.path.join(folder_path, png_file)
            dst_file = os.path.join(output_path, png_file)
            shutil.copy2(src_file, dst_file)
            print(f"\rCopied: {src_file} → {dst_file}", flush=True)
        
        print(f"Created folder {folder_name} with {len(png_files)} images from {folder_path}")

def full_flat(rootFolder, outputFolder):
    os.makedirs(outputFolder, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(rootFolder):
        for filename in filenames:
            if filename.endswith('.png'):
                fullPath = os.path.join(dirpath, filename)
                relPath = os.path.relpath(fullPath, rootFolder)
                newName = relPath.replace(os.sep, '_')
                outputPath = os.path.join(outputFolder, newName)
                shutil.copy2(fullPath, outputPath)
                print(f"Copied: {fullPath} → {outputPath}", flush=True)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python flat_dir.py <source_root_path> <output_folder_path>")
        sys.exit(1)

    sourceRoot = sys.argv[1]
    outputFolder = sys.argv[2]
    method = sys.argv[3]

    if method == '-f':
        full_flat(sourceRoot, outputFolder)
    else:
        flat_with_leaf(sourceRoot, outputFolder)
    print(f"Done! Organized PNG files into sequential folders in {outputFolder}")