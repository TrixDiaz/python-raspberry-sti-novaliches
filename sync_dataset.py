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

def download_image(url, save_path):
    """Download a single image from URL"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
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
