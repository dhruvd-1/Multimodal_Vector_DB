"""
Clean Python files by removing emojis and excessive comments
"""
import os
import re

FILES_TO_CLEAN = [
    "search_cross_modal.py",
    "search_images.py",
    "search_videos.py",
    "search_audio.py",
    "search_text.py",
    "build_cross_modal_index.py",
    "build_all_indices.py",
    "benchmark_cross_modal.py",
]

def remove_emojis(text):
    """Remove emoji characters from text"""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE
    )
    return emoji_pattern.sub('', text)

def clean_file(filepath):
    """Clean a single Python file"""
    if not os.path.exists(filepath):
        print(f"Skipping {filepath} (not found)")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        line_no_emoji = remove_emojis(line)

        stripped = line_no_emoji.strip()
        if stripped.startswith('#') and not stripped.startswith('#!'):
            continue

        cleaned_lines.append(line_no_emoji)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

    print(f"Cleaned {filepath}")

def main():
    print("Cleaning Python files...")
    print("=" * 60)

    for filename in FILES_TO_CLEAN:
        clean_file(filename)

    print("=" * 60)
    print("Done!")

if __name__ == "__main__":
    main()
