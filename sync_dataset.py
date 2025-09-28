#!/usr/bin/env python3
"""
Dataset Sync Script
Downloads all user images from the backend API and organizes them in dataset folders
"""

import requests
import os
import json
from urllib.parse import urlparse
import time

# Backend API configuration
BACKEND_BASE_URL = "http://localhost:3000"  # Update this to your backend URL
API_ENDPOINT = f"{BACKEND_BASE_URL}/api/faces/dataset-sync"  # Public dataset sync endpoint

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

def sync_user_images():
    """Download all user images and organize them in dataset folders"""
    print("Starting dataset sync...")
    
    # Create dataset structure
    create_dataset_structure()
    
    try:
        # Get all users' face data from backend (public endpoint)
        print(f"Fetching user data from: {API_ENDPOINT}")
        response = requests.get(API_ENDPOINT, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('success'):
            print(f"API Error: {data.get('message', 'Unknown error')}")
            return False
        
        users_data = data.get('data', {}).get('faces', [])
        print(f"Found {len(users_data)} users with face data")
        
        if len(users_data) == 0:
            print("No users with face data found in database.")
            return False
        
        total_images = 0
        successful_downloads = 0
        
        for user_data in users_data:
            user_id = user_data.get('userId')
            user_name = user_data.get('name', f'user_{user_id}')
            images = user_data.get('images', [])
            
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
    
    # Check if backend is accessible
    try:
        response = requests.get(f"{BACKEND_BASE_URL}/health", timeout=5)
        print("✓ Backend server is accessible")
    except:
        print("⚠ Warning: Cannot reach backend server")
        print(f"Make sure your backend is running at: {BACKEND_BASE_URL}")
        print("Backend should be running on the port shown in your backend console")
        print()
    
    # Sync dataset
    if sync_user_images():
        print("\n✓ Dataset sync completed successfully")
        print("✓ User images downloaded and organized in dataset folders")
        print("✓ Face recognition dataset is ready!")
    else:
        print("\n✗ Dataset sync failed")
        print("Make sure:")
        print("1. Your backend server is running")
        print("2. You have users with face images in your database")
        print("3. The backend can connect to your Neon database")
        return False
    
    print("\n=== Sync Complete ===")
    print("Dataset folders created with user images!")
    return True

if __name__ == "__main__":
    main()
