import os
import shutil

src_dir_path = "/home/atiq/Desktop/try-on/kaggel/archive(2)/clothes/test/"
dest_dir_path = "/home/atiq/Desktop/try-on/kaggel/diff_dress/"

os.makedirs(dest_dir_path, exist_ok=True)

for root, sub, files in os.walk(src_dir_path):
    folder_name = os.path.basename(root)
    # print(root)
    
    for file in files:
        src_path = os.path.join(root, file)
        dest_path = os.path.join(dest_dir_path, file)
        
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(file)
            
            count = 0
            
            while os.path.exists(dest_path):
                dest_path = os.path.join(dest_dir_path, f"{folder_name}_{count}{ext}")
                count += 1
                
        shutil.copy2(src_path, dest_path)
        
print(f"Copied all files from {src_dir_path} to {dest_dir_path}")

# import os
# import shutil

# source_dir = "/home/atiq/Desktop/try-on/kaggel/archive(2)/clothes/train/"
# destination_dir = "/home/atiq/Desktop/try-on/kaggel/diff_dress/"

# # Create destination folder if it doesn't exist
# os.makedirs(destination_dir, exist_ok=True)

# for root, dirs, files in os.walk(source_dir):
#     for file in files:
#         src_path = os.path.join(root, file)
#         dest_path = os.path.join(destination_dir, file)

#         # Avoid overwriting files with same name
#         if os.path.exists(dest_path):
#             base, ext = os.path.splitext(file)
#             count = 1
#             while os.path.exists(dest_path):
#                 dest_path = os.path.join(
#                     destination_dir, f"{base}_{count}{ext}"
#                 )
#                 count += 1

#         shutil.copy2(src_path, dest_path)  # copy2 keeps metadata

# print("✅ All files copied successfully!")
