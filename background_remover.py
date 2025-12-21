"""
BACKGROUND REMOVER FOR FACE RECOGNITION DATASET
Menggunakan Remove.bg API untuk menghilangkan background dari dataset

Benefit:
- Fokus pada wajah, eliminasi background noise
- Meningkatkan akurasi model
- Konsistensi background (semua transparan/putih)
"""

import os
import requests
import time
from pathlib import Path
import shutil

# ============================================================================
# CONFIGURATION
# ============================================================================

REMOVE_BG_API_KEY = "enhSGdvZYnVDcWzp9sFBSR2N"
REMOVE_BG_URL = "https://api.remove.bg/v1.0/removebg"

# ============================================================================
# BACKGROUND REMOVAL FUNCTIONS
# ============================================================================

def remove_background_from_image(image_path, output_path, api_key):
    """
    Remove background dari satu gambar menggunakan Remove.bg API
    
    Args:
        image_path: Path ke gambar input
        output_path: Path untuk save hasil
        api_key: Remove.bg API key
    
    Returns:
        True jika sukses, False jika gagal
    """
    
    try:
        # Read image file
        with open(image_path, 'rb') as image_file:
            # Prepare request
            response = requests.post(
                REMOVE_BG_URL,
                files={'image_file': image_file},
                data={
                    'size': 'auto',  # Original size
                    'bg_color': 'white'  # White background (bisa 'transparent' juga)
                },
                headers={'X-Api-Key': api_key}
            )
        
        # Check response
        if response.status_code == requests.codes.ok:
            # Save result
            with open(output_path, 'wb') as out_file:
                out_file.write(response.content)
            return True
        else:
            print(f"   [ERROR] Status {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"   [ERROR] Exception: {str(e)}")
        return False


def process_dataset(input_folder='dataset', output_folder='dataset_nobg', 
                   backup_folder='dataset_backup', api_key=REMOVE_BG_API_KEY):
    """
    Process semua gambar dalam dataset folder
    
    Args:
        input_folder: Folder dataset original
        output_folder: Folder untuk hasil (background removed)
        backup_folder: Folder backup dataset original
        api_key: Remove.bg API key
    """
    
    print("=" * 70)
    print("        BACKGROUND REMOVAL FOR FACE RECOGNITION DATASET")
    print("=" * 70)
    
    # Check if dataset exists
    if not os.path.exists(input_folder):
        print(f"\n[X] ERROR: Folder '{input_folder}' tidak ditemukan!")
        print("Pastikan Anda sudah mengumpulkan dataset terlebih dahulu.")
        return
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [
        f for f in os.listdir(input_folder) 
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    if len(image_files) == 0:
        print(f"\n[X] ERROR: Tidak ada gambar di folder '{input_folder}'!")
        return
    
    print(f"\n[OK] Ditemukan {len(image_files)} gambar dalam dataset")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    print(f"[OK] Output folder: '{output_folder}'")
    
    # Ask for confirmation
    print("\n" + "-" * 70)
    print("PERINGATAN:")
    print("-" * 70)
    print(f"Proses ini akan:")
    print(f"  1. Backup dataset original ke '{backup_folder}'")
    print(f"  2. Remove background dari {len(image_files)} gambar")
    print(f"  3. Save hasil ke '{output_folder}'")
    print(f"  4. API calls: {len(image_files)} (Remove.bg API limit: 50/month free)")
    print("\nNote: Setiap API call menggunakan 1 credit dari quota Anda.")
    print(f"      Total yang dibutuhkan: {len(image_files)} credits")
    
    # Check API quota warning
    if len(image_files) > 50:
        print(f"\n[WARNING] Dataset Anda punya {len(image_files)} gambar.")
        print("          Free plan Remove.bg hanya 50 calls/month!")
        print("          Consider:")
        print("          1. Proses sebagian dataset dulu")
        print("          2. Upgrade ke paid plan")
        print("          3. Gunakan alternative (rembg library - offline)")
    
    confirm = input("\nLanjutkan? (y/n): ").strip().lower()
    if confirm != 'y':
        print("\n[INFO] Proses dibatalkan.")
        return
    
    # Backup original dataset
    print("\n" + "=" * 70)
    print("STEP 1: BACKUP DATASET ORIGINAL")
    print("=" * 70)
    
    if os.path.exists(backup_folder):
        print(f"[INFO] Backup folder sudah ada: '{backup_folder}'")
        overwrite = input("Overwrite backup? (y/n): ").strip().lower()
        if overwrite == 'y':
            shutil.rmtree(backup_folder)
            shutil.copytree(input_folder, backup_folder)
            print(f"[OK] Backup updated ke '{backup_folder}'")
        else:
            print(f"[OK] Menggunakan backup existing")
    else:
        shutil.copytree(input_folder, backup_folder)
        print(f"[OK] Dataset di-backup ke '{backup_folder}'")
    
    # Process images
    print("\n" + "=" * 70)
    print("STEP 2: REMOVING BACKGROUNDS")
    print("=" * 70)
    
    success_count = 0
    failed_count = 0
    
    for idx, filename in enumerate(image_files, 1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        print(f"\n[{idx}/{len(image_files)}] Processing: {filename}")
        
        # Remove background
        success = remove_background_from_image(input_path, output_path, api_key)
        
        if success:
            print(f"   [OK] Background removed successfully")
            success_count += 1
        else:
            print(f"   [FAILED] Could not remove background")
            failed_count += 1
            # Copy original if failed
            shutil.copy2(input_path, output_path)
            print(f"   [INFO] Copied original file as fallback")
        
        # Delay to respect API rate limits
        if idx < len(image_files):  # Don't delay after last image
            time.sleep(1)  # 1 second delay between requests
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total gambar: {len(image_files)}")
    print(f"Berhasil:     {success_count}")
    print(f"Gagal:        {failed_count}")
    print(f"Success rate: {(success_count/len(image_files)*100):.1f}%")
    
    print(f"\nHasil tersimpan di: '{output_folder}'")
    print(f"Backup original di: '{backup_folder}'")
    
    # Next steps
    print("\n" + "=" * 70)
    print("LANGKAH SELANJUTNYA")
    print("=" * 70)
    print("""
1. VERIFIKASI HASIL:
   - Buka folder 'dataset_nobg'
   - Periksa apakah background sudah terhapus dengan baik
   - Pastikan wajah masih utuh dan jelas

2. UPDATE DATASET:
   Jika hasil memuaskan, ganti dataset dengan yang baru:
   
   Option A (Rename):
   - Rename 'dataset' menjadi 'dataset_old'
   - Rename 'dataset_nobg' menjadi 'dataset'
   
   Option B (Replace):
   - Hapus isi folder 'dataset'
   - Copy semua dari 'dataset_nobg' ke 'dataset'

3. RETRAIN MODEL:
   python 1_train.py
   
   Model akan belajar dari gambar dengan background yang sudah dihapus.

4. TEST RECOGNITION:
   python maincam.py
   
   Lihat apakah akurasi meningkat!

EXPECTED IMPROVEMENTS:
- Akurasi bisa meningkat 2-5%
- Model fokus pada fitur wajah, bukan background
- Lebih robust terhadap variasi background saat testing
- Confusion matrix lebih baik untuk kelas yang mirip
""")


def check_api_quota(api_key):
    """
    Check remaining API quota untuk Remove.bg
    """
    print("=" * 70)
    print("CHECK REMOVE.BG API QUOTA")
    print("=" * 70)
    
    # Remove.bg doesn't have direct quota check endpoint
    # But we can make a test call and check headers
    
    print(f"\nAPI Key: {api_key[:10]}...{api_key[-4:]}")
    print("\nNote: Remove.bg free plan:")
    print("  - 50 API calls per month")
    print("  - Resets monthly")
    print("  - Preview API calls don't count")
    
    print("\nUntuk check quota detail, login ke:")
    print("  https://www.remove.bg/users/sign_in")
    print("  Dashboard akan menampilkan remaining credits")


def process_selective(input_folder='dataset', output_folder='dataset_nobg',
                     max_images=None, api_key=REMOVE_BG_API_KEY):
    """
    Process hanya sebagian dataset (useful untuk free plan limit)
    
    Args:
        input_folder: Folder dataset
        output_folder: Folder output
        max_images: Maximum number of images to process (None = all)
        api_key: API key
    """
    
    print("=" * 70)
    print("        SELECTIVE BACKGROUND REMOVAL")
    print("=" * 70)
    
    if not os.path.exists(input_folder):
        print(f"\n[X] ERROR: Folder '{input_folder}' tidak ditemukan!")
        return
    
    # Get all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    all_images = [
        f for f in os.listdir(input_folder)
        if os.path.splitext(f.lower())[1] in image_extensions
    ]
    
    if not all_images:
        print(f"\n[X] ERROR: Tidak ada gambar di '{input_folder}'!")
        return
    
    print(f"\n[OK] Total gambar di dataset: {len(all_images)}")
    
    # Determine how many to process
    if max_images is None or max_images > len(all_images):
        images_to_process = all_images
    else:
        images_to_process = all_images[:max_images]
    
    print(f"[OK] Akan diproses: {len(images_to_process)} gambar")
    
    # Group by person (for balanced processing)
    person_groups = {}
    for img in images_to_process:
        try:
            person_name = img.split('_')[0]
            if person_name not in person_groups:
                person_groups[person_name] = []
            person_groups[person_name].append(img)
        except:
            pass
    
    print(f"[OK] Ditemukan {len(person_groups)} orang dalam dataset")
    for person, imgs in person_groups.items():
        print(f"     - {person}: {len(imgs)} gambar")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Process
    print("\n" + "=" * 70)
    print("PROCESSING...")
    print("=" * 70)
    
    success = 0
    failed = 0
    
    for idx, filename in enumerate(images_to_process, 1):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        print(f"\n[{idx}/{len(images_to_process)}] {filename}")
        
        if remove_background_from_image(input_path, output_path, api_key):
            print(f"   [OK] Success")
            success += 1
        else:
            print(f"   [FAILED]")
            failed += 1
            shutil.copy2(input_path, output_path)
        
        time.sleep(1)
    
    print("\n" + "=" * 70)
    print(f"Done! Success: {success}, Failed: {failed}")
    print("=" * 70)


# ============================================================================
# ALTERNATIVE: OFFLINE BACKGROUND REMOVAL (NO API NEEDED)
# ============================================================================

def setup_offline_remover():
    """
    Setup untuk background removal offline menggunakan rembg library
    (Tidak perlu API key, unlimited, tapi perlu install library)
    """
    
    print("=" * 70)
    print("ALTERNATIVE: OFFLINE BACKGROUND REMOVAL")
    print("=" * 70)
    
    print("""
Jika Anda ingin remove background tanpa API (unlimited, offline):

1. INSTALL REMBG:
   pip install rembg[gpu]  # Jika punya GPU
   atau
   pip install rembg       # CPU only
   
2. DOWNLOAD MODEL (first time only):
   Model akan auto-download saat first run (~180MB)
   
3. USE CODE:
   
   from rembg import remove
   from PIL import Image
   import os
   
   input_folder = 'dataset'
   output_folder = 'dataset_nobg'
   
   os.makedirs(output_folder, exist_ok=True)
   
   for filename in os.listdir(input_folder):
       if filename.endswith(('.jpg', '.png')):
           input_path = os.path.join(input_folder, filename)
           output_path = os.path.join(output_folder, filename)
           
           # Remove background
           with open(input_path, 'rb') as inp:
               with open(output_path, 'wb') as out:
                   input_data = inp.read()
                   output_data = remove(input_data)
                   out.write(output_data)
           
           print(f"Processed: {filename}")

KEUNTUNGAN REMBG:
- Unlimited (no API quota)
- Offline (no internet needed)
- Free forever
- Good quality
- Fast with GPU

KEKURANGAN:
- Perlu install library (~180MB)
- Lebih lambat di CPU
- Kualitas sedikit di bawah Remove.bg
""")


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Main menu untuk background removal tools"""
    
    while True:
        print("\n" + "=" * 70)
        print("        BACKGROUND REMOVAL TOOLS")
        print("=" * 70)
        print("\nOptions:")
        print("  1. Remove background dari SEMUA dataset (Remove.bg API)")
        print("  2. Remove background SELECTIVE (limit jumlah)")
        print("  3. Check API quota info")
        print("  4. Info: Alternative offline method (unlimited)")
        print("  0. Exit")
        print("-" * 70)
        
        choice = input("\nPilih (0-4): ").strip()
        
        if choice == '1':
            process_dataset()
        
        elif choice == '2':
            try:
                max_img = int(input("\nBerapa gambar yang ingin diproses? (max 50): "))
                process_selective(max_images=max_img)
            except ValueError:
                print("[ERROR] Input harus angka!")
        
        elif choice == '3':
            check_api_quota(REMOVE_BG_API_KEY)
        
        elif choice == '4':
            setup_offline_remover()
        
        elif choice == '0':
            print("\nGoodbye!")
            break
        
        else:
            print("\n[ERROR] Pilihan tidak valid!")


if __name__ == "__main__":
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*18 + "BACKGROUND REMOVER" + " "*31 + "║")
    print("║" + " "*15 + "Face Recognition Dataset" + " "*28 + "║")
    print("╚" + "="*68 + "╝")
    
    main()