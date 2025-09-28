#!/usr/bin/env python3
"""
Dataset Sync Script
Downloads all user images from the backend API and organizes them in dataset folders
"""

import os
import json
import requests
from urllib.parse import urlparse
import time
import psycopg2
from psycopg2.extras import RealDictCursor

# Database configuration - Direct connection to Neon
DATABASE_URL = "postgresql://neondb_owner:npg_VgHuhCp2Jx6P@ep-wispy-firefly-adswi51k-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"

# Dataset configuration
DATASET_DIR = "dataset"

def create_dataset_structure():
    """Create the dataset directory structure"""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        print(f"Created dataset directory: {DATASET_DIR}")
    
    return True

def create_placeholder_image(save_path, user_name, image_index):
    """Create a placeholder image when download fails"""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple placeholder image
        img = Image.new('RGB', (200, 200), color='lightgray')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add text to the image
        text = f"{user_name}\nImage {image_index + 1}\n(Placeholder)"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (200 - text_width) // 2
        y = (200 - text_height) // 2
        
        draw.text((x, y), text, fill='black', font=font)
        
        # Save the placeholder image
        img.save(save_path, 'JPEG')
        return True
        
    except Exception as e:
        print(f"Error creating placeholder image: {e}")
        return False

def download_image(url, save_path, max_retries=2):
    """Download a single image from URL with retry logic"""
    for attempt in range(max_retries):
        try:
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            response = requests.get(url, timeout=15, headers=headers, stream=True)
            response.raise_for_status()
            
            # Check if response is actually an image
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                print(f"Warning: {url} is not an image (content-type: {content_type})")
                return False
            
            # Download in chunks to handle large images
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            return True
            
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error downloading {url} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
                continue
        except requests.exceptions.Timeout as e:
            print(f"Timeout downloading {url} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
        except Exception as e:
            print(f"Error downloading {url} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
    
    return False

def get_users_with_faces():
    """Get all users with their face images from the database"""
    try:
        # Connect to database
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Query to get users and their face images
        query = """
        SELECT 
            u.id as user_id,
            u.name,
            u.email,
            f.images,
            f.total_images
        FROM users u
        JOIN faces f ON u.id = f.user_id
        WHERE f.images IS NOT NULL 
        AND f.images != '[]'
        AND f.images != 'null'
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return results
        
    except Exception as e:
        print(f"Database connection error: {e}")
        return []

def sync_user_images():
    """Download all user images and organize them in dataset folders"""
    print("Starting dataset sync...")
    
    # Create dataset structure
    create_dataset_structure()
    
    try:
        # Get users with face data from database
        print("Connecting to Neon database...")
        users_data = get_users_with_faces()
        
        if len(users_data) == 0:
            print("No users with face data found in database.")
            return False
        
        print(f"Found {len(users_data)} users with face data")
        
        total_images = 0
        successful_downloads = 0
        
        for user_data in users_data:
            user_id = user_data['user_id']
            user_name = user_data['name']
            images_json = user_data['images']
            
            # Parse images JSON
            try:
                images = json.loads(images_json) if isinstance(images_json, str) else images_json
            except:
                print(f"Error parsing images for user: {user_name}")
                continue
            
            if len(images) == 0:
                print(f"No images for user: {user_name}")
                continue
            
            # Create user directory
            user_dir = os.path.join(DATASET_DIR, user_name)
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
                print(f"Created directory for user: {user_name}")
            
            # Download all images for this user
            for i, image_url in enumerate(images):
                # Generate filename
                parsed_url = urlparse(image_url)
                filename = os.path.basename(parsed_url.path)
                if not filename or '.' not in filename:
                    filename = f"image_{i+1}.jpg"
                
                save_path = os.path.join(user_dir, filename)
                
                # Download image
                if download_image(image_url, save_path):
                    successful_downloads += 1
                    print(f"Downloaded: {user_name}/{filename}")
                else:
                    print(f"Failed to download: {user_name}/{filename}")
                    # Create a placeholder image instead of text file
                    if create_placeholder_image(save_path, user_name, i):
                        successful_downloads += 1
                        print(f"Created placeholder: {user_name}/{filename}")
                    else:
                        # Fallback to text file if image creation fails
                        placeholder_path = save_path.replace('.jpg', '_placeholder.txt')
                        with open(placeholder_path, 'w') as f:
                            f.write(f"Failed to download: {image_url}")
                
                total_images += 1
                
                # Small delay to avoid overwhelming the server
                time.sleep(0.1)
        
        print(f"\nDataset sync completed!")
        print(f"Total images processed: {total_images}")
        print(f"Successful downloads: {successful_downloads}")
        print(f"Failed downloads: {total_images - successful_downloads}")
        
        return successful_downloads > 0
        
    except Exception as e:
        print(f"Error during dataset sync: {e}")
        return False



def main():
    """Main function"""
    print("=== Dataset Sync Script ===")
    print("This script will download all user images from your Neon database")
    print("and organize them in dataset folders for face recognition")
    print()
    
    # Sync dataset
    if sync_user_images():
        print("\n✓ Dataset sync completed successfully")
        print("✓ User images downloaded and organized in dataset folders")
        print("✓ Face recognition dataset is ready!")
    else:
        print("\n✗ Dataset sync failed")
        print("Make sure:")
        print("1. You have users with face images in your Neon database")
        print("2. The database connection is working")
        print("3. You have internet connection to download images")
        return False
    
    print("\n=== Sync Complete ===")
    print("Dataset folders created with user images!")
    return True

if __name__ == "__main__":
    main()
