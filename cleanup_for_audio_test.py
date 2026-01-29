"""
Quick disk space cleanup script to free space for CLAP model download.
This will clean up pip cache to free ~700MB.
"""

import subprocess
import os

def get_free_space_gb(drive='C:'):
    """Get free space on drive in GB."""
    import shutil
    stats = shutil.disk_usage(drive)
    return stats.free / (1024**3)

def main():
    print("=" * 60)
    print("DISK SPACE CLEANUP FOR CLAP MODEL")
    print("=" * 60)
    
    # Check current free space
    free_space = get_free_space_gb('C:')
    print(f"\nCurrent free space on C: drive: {free_space:.2f} GB")
    
    if free_space < 2.0:
        print(f"⚠️  Need at least 2 GB free for CLAP model")
        print(f"   Currently only {free_space:.2f} GB available\n")
        
        # Clean pip cache
        print("Cleaning pip cache...")
        try:
            result = subprocess.run(['pip', 'cache', 'purge'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("✓ Pip cache cleaned")
            else:
                print(f"✗ Failed to clean pip cache: {result.stderr}")
        except Exception as e:
            print(f"✗ Error cleaning pip cache: {e}")
        
        # Check new free space
        new_free_space = get_free_space_gb('C:')
        freed = new_free_space - free_space
        print(f"\n✓ Freed {freed:.2f} GB")
        print(f"  New free space: {new_free_space:.2f} GB")
        
        if new_free_space >= 2.0:
            print("\n✓ Enough space available now!")
            print("  Run: python scripts/test_audio_embedder.py")
        else:
            print(f"\n⚠️  Still need {2.0 - new_free_space:.2f} GB more")
            print("\nAdditional cleanup suggestions:")
            print("  1. Run Disk Cleanup: cleanmgr /d C:")
            print("  2. Empty Recycle Bin")
            print("  3. Remove large unused files")
            print("  4. Move OneDrive files to online-only")
    else:
        print("✓ Sufficient disk space available!")
        print("  Run: python scripts/test_audio_embedder.py")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
