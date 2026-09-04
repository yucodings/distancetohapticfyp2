import os
import zipfile
import sys

# ============================================================
# CONFIGURATION
# ============================================================

folder_to_zip = "4.0_realsense"  # Replace with the folder you want to zip

# ============================================================


folder_to_zip = os.path.abspath(folder_to_zip)

if not os.path.isdir(folder_to_zip):
    print(f"❌ Folder does not exist: {folder_to_zip}")
    sys.exit(1)


folder_name = os.path.basename(folder_to_zip)
parent_directory = os.path.dirname(folder_to_zip)

zip_file_path = os.path.join(
    parent_directory,
    folder_name + ".zip"
)


# ============================================================
# FIND ALL FILES FIRST
# ============================================================

all_files = []

for root, dirs, files in os.walk(folder_to_zip):
    for file in files:
        file_path = os.path.join(root, file)
        all_files.append(file_path)


total_files = len(all_files)

# Calculate total size
total_size = sum(
    os.path.getsize(file)
    for file in all_files
)

total_size_mb = total_size / (1024 * 1024)


print("=" * 70)
print("ZIP FOLDER")
print("=" * 70)

print(f"📁 Source folder : {folder_to_zip}")
print(f"📦 ZIP file name : {os.path.basename(zip_file_path)}")
print(f"📍 ZIP exact path: {zip_file_path}")
print(f"📄 Total files   : {total_files}")
print(f"💾 Total size    : {total_size_mb:.2f} MB")

print("=" * 70)
print()


# ============================================================
# PROGRESS BAR FUNCTION
# ============================================================

def show_progress(current, total, bar_length=40):

    if total == 0:
        percent = 100
    else:
        percent = (current / total) * 100

    filled_length = int(bar_length * current / total) if total else bar_length

    bar = "█" * filled_length
    bar += "-" * (bar_length - filled_length)

    print(
        f"\r[{bar}] "
        f"{percent:6.2f}% "
        f"({current}/{total} files)",
        end="",
        flush=True
    )


# ============================================================
# CREATE ZIP
# ============================================================

with zipfile.ZipFile(
    zip_file_path,
    "w",
    zipfile.ZIP_DEFLATED,
    allowZip64=True
) as zipf:

    for index, file_path in enumerate(all_files, start=1):

        # Path stored inside ZIP
        relative_path = os.path.relpath(
            file_path,
            parent_directory
        )

        zipf.write(
            file_path,
            arcname=relative_path
        )

        show_progress(index, total_files)


print("\n")


# ============================================================
# FINAL RESULT
# ============================================================

zip_size = os.path.getsize(zip_file_path)
zip_size_mb = zip_size / (1024 * 1024)

print("=" * 70)
print("✅ ZIP COMPLETED")
print("=" * 70)

print(f"📁 Source folder : {folder_to_zip}")
print(f"📄 Files zipped  : {total_files}")
print(f"💾 Original size : {total_size_mb:.2f} MB")
print(f"📦 ZIP file name : {os.path.basename(zip_file_path)}")
print(f"📦 ZIP size      : {zip_size_mb:.2f} MB")
print(f"📍 ZIP exact path: {zip_file_path}")

print("=" * 70)
