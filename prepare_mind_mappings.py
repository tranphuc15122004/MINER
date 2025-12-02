"""
Script to generate user2id.json and category2id.json from MIND dataset
"""
import json
import csv
from pathlib import Path
from collections import Counter


def create_user2id(behaviors_paths):
    """
    Create user2id mapping from behaviors.tsv files
    
    Args:
        behaviors_paths: List of paths to behaviors.tsv files
    """
    users = set()
    
    for behaviors_path in behaviors_paths:
        print(f"Reading users from {behaviors_path}...")
        with open(behaviors_path, 'r', encoding='utf-8', newline='') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            for line in tsv_reader:
                if len(line) >= 2:
                    user_id = line[1]  # User ID is in column 1
                    users.add(user_id)
    
    # Create mapping with special tokens
    user2id = {
        'pad': 0,
        'unk': 1
    }
    
    # Add all users
    for idx, user in enumerate(sorted(users), start=2):
        user2id[user] = idx
    
    print(f"Total users: {len(users)}")
    print(f"Total entries (with pad/unk): {len(user2id)}")
    
    return user2id


def create_category2id(news_paths, use_subcategory=False):
    """
    Create category2id mapping from news.tsv files
    
    Args:
        news_paths: List of paths to news.tsv files
        use_subcategory: If True, use subcategory (column 2), else use main category (column 1)
    """
    categories = set()
    
    for news_path in news_paths:
        print(f"Reading categories from {news_path}...")
        with open(news_path, 'r', encoding='utf-8', newline='') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            for line in tsv_reader:
                if len(line) >= 3:
                    if use_subcategory:
                        category = line[2]  # Subcategory is in column 2
                    else:
                        category = line[1]  # Main category is in column 1
                    categories.add(category)
    
    # Create mapping with special tokens
    category2id = {
        'pad': 0,
        'unk': 1
    }
    
    # Add all categories
    for idx, category in enumerate(sorted(categories), start=2):
        category2id[category] = idx
    
    print(f"Total categories: {len(categories)}")
    print(f"Categories found: {sorted(categories)}")
    print(f"Total entries (with pad/unk): {len(category2id)}")
    
    return category2id


def main():
    # Paths to MIND dataset files
    # Adjust these paths based on where you downloaded MIND dataset
    train_behaviors = Path('data/train/behaviors.tsv')
    train_news = Path('data/train/news.tsv')
    valid_behaviors = Path('data/valid/behaviors.tsv')
    valid_news = Path('data/valid/news.tsv')
    
    # Check if files exist
    for file_path in [train_behaviors, train_news, valid_behaviors, valid_news]:
        if not file_path.exists():
            print(f"WARNING: {file_path} does not exist!")
            print(f"Please download MIND dataset and place files in the correct location.")
            return
    
    print("=" * 60)
    print("Creating user2id.json...")
    print("=" * 60)
    user2id = create_user2id([train_behaviors, valid_behaviors])
    
    print("\n" + "=" * 60)
    print("Creating category2id.json...")
    print("=" * 60)
    
    # Change use_subcategory to True if you want to use subcategories instead
    use_subcategory = False  # Set to True for subcategories (more granular)
    print(f"Using {'subcategories' if use_subcategory else 'main categories'}")
    
    category2id = create_category2id([train_news, valid_news], use_subcategory=use_subcategory)
    
    # Save to JSON files
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    
    user2id_path = output_dir / 'user2id.json'
    category2id_path = output_dir / 'category2id.json'
    
    with open(user2id_path, 'w', encoding='utf-8') as f:
        json.dump(user2id, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved user2id.json to {user2id_path}")
    
    with open(category2id_path, 'w', encoding='utf-8') as f:
        json.dump(category2id, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved category2id.json to {category2id_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total users: {len(user2id) - 2} (+ pad, unk)")
    print(f"Total categories: {len(category2id) - 2} (+ pad, unk)")
    print("\nFiles created successfully!")


if __name__ == '__main__':
    main()
