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
API_ENDPOINT = f"{BACKEND_BASE_URL}/api/faces/all-users"  # You'll need to create this endpoint

# Dataset configuration
DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pickle"

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
        # Get all users' face data from backend
        print(f"Fetching user data from: {API_ENDPOINT}")
        response = requests.get(API_ENDPOINT, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get('success'):
            print(f"API Error: {data.get('message', 'Unknown error')}")
            return False
        
        users_data = data.get('data', {}).get('faces', [])
        print(f"Found {len(users_data)} users with face data")
        
        total_images = 0
        successful_downloads = 0
        
        for user_data in users_data:
            user_id = user_data.get('userId')
            user_name = user_data.get('name', f'user_{user_id}')
            total_images_count = user_data.get('totalImages', 0)
            
            if total_images_count == 0:
                print(f"No images for user: {user_name}")
                continue
            
            # Create user directory
            user_dir = os.path.join(DATASET_DIR, user_name)
            if not os.path.exists(user_dir):
                os.makedirs(user_dir)
                print(f"Created directory for user: {user_name}")
            
            # Get user's images (you'll need to implement this endpoint)
            user_images_url = f"{BACKEND_BASE_URL}/api/faces/user-images/{user_id}"
            try:
                user_response = requests.get(user_images_url, timeout=30)
                user_response.raise_for_status()
                user_data = user_response.json()
                
                if user_data.get('success'):
                    images = user_data.get('data', {}).get('faceRecord', {}).get('images', [])
                    
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
                
            except Exception as e:
                print(f"Error fetching images for user {user_name}: {e}")
                continue
        
        print(f"\nDataset sync completed!")
        print(f"Total images processed: {total_images}")
        print(f"Successful downloads: {successful_downloads}")
        print(f"Failed downloads: {total_images - successful_downloads}")
        
        return successful_downloads > 0
        
    except Exception as e:
        print(f"Error during dataset sync: {e}")
        return False

def generate_encodings():
    """Generate encodings.pickle from the dataset"""
    try:
        import face_recognition
        import pickle
        
        print("Generating face encodings...")
        
        known_encodings = []
        known_names = []
        
        # Walk through dataset directory
        for root, dirs, files in os.walk(DATASET_DIR):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    image_path = os.path.join(root, file)
                    user_name = os.path.basename(root)
                    
                    try:
                        # Load image and find face encodings
                        image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image)
                        
                        for encoding in face_encodings:
                            known_encodings.append(encoding)
                            known_names.append(user_name)
                            print(f"Processed: {user_name}/{file}")
                            
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                        continue
        
        # Save encodings to pickle file
        encodings_data = {
            'encodings': known_encodings,
            'names': known_names
        }
        
        with open(ENCODINGS_FILE, 'wb') as f:
            pickle.dump(encodings_data, f)
        
        print(f"Generated encodings.pickle with {len(known_encodings)} face encodings")
        print(f"Unique users: {len(set(known_names))}")
        
        return True
        
    except ImportError:
        print("Error: face_recognition library not installed")
        print("Run: pip install face-recognition")
        return False
    except Exception as e:
        print(f"Error generating encodings: {e}")
        return False

def main():
    """Main function"""
    print("=== Dataset Sync Script ===")
    print("This script will download all user images and generate face encodings")
    print()
    
    # Check if backend is accessible
    try:
        response = requests.get(f"{BACKEND_BASE_URL}/api/health", timeout=5)
        print("✓ Backend server is accessible")
    except:
        print("⚠ Warning: Cannot reach backend server")
        print(f"Make sure your backend is running at: {BACKEND_BASE_URL}")
        print()
    
    # Sync dataset
    if sync_user_images():
        print("\n✓ Dataset sync completed successfully")
        
        # Generate encodings
        if generate_encodings():
            print("✓ Face encodings generated successfully")
            print(f"✓ Encodings saved to: {ENCODINGS_FILE}")
        else:
            print("✗ Failed to generate face encodings")
    else:
        print("\n✗ Dataset sync failed")
        return False
    
    print("\n=== Sync Complete ===")
    print("You can now use the face recognition system!")
    return True

if __name__ == "__main__":
    main()
