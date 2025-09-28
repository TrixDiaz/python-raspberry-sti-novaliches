import os
from imutils import paths
import face_recognition
import pickle
import cv2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(dataset_path="dataset", output_file="encodings.pickle"):
    """
    Train the face recognition model using images from the dataset.
    
    Args:
        dataset_path: Path to the dataset directory
        output_file: Output file for the trained encodings
        
    Returns:
        bool: True if training was successful, False otherwise
    """
    try:
        logger.info("Starting face recognition model training...")
        
        # Check if dataset directory exists
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset directory not found: {dataset_path}")
            return False
        
        # Get all image paths
        imagePaths = list(paths.list_images(dataset_path))
        if not imagePaths:
            logger.warning(f"No images found in dataset directory: {dataset_path}")
            return False
        
        logger.info(f"Found {len(imagePaths)} images to process")
        
        knownEncodings = []
        knownNames = []
        processed_count = 0
        failed_count = 0
        
        for (i, imagePath) in enumerate(imagePaths):
            try:
                logger.info(f"Processing image {i + 1}/{len(imagePaths)}: {imagePath}")
                name = imagePath.split(os.path.sep)[-2]
                
                # Load and process image
                image = cv2.imread(imagePath)
                if image is None:
                    logger.warning(f"Could not load image: {imagePath}")
                    failed_count += 1
                    continue
                
                # Convert BGR to RGB
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect face locations
                boxes = face_recognition.face_locations(rgb, model="hog")
                if not boxes:
                    logger.warning(f"No faces detected in: {imagePath}")
                    failed_count += 1
                    continue
                
                # Get face encodings
                encodings = face_recognition.face_encodings(rgb, boxes)
                if not encodings:
                    logger.warning(f"No face encodings generated for: {imagePath}")
                    failed_count += 1
                    continue
                
                # Add encodings and names
                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(name)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing image {imagePath}: {str(e)}")
                failed_count += 1
                continue
        
        if not knownEncodings:
            logger.error("No face encodings were generated. Training failed.")
            return False
        
        logger.info(f"Training summary:")
        logger.info(f"  - Successfully processed: {processed_count} images")
        logger.info(f"  - Failed to process: {failed_count} images")
        logger.info(f"  - Total face encodings: {len(knownEncodings)}")
        logger.info(f"  - Unique faces: {len(set(knownNames))}")
        
        # Serialize encodings
        logger.info("Serializing encodings...")
        data = {"encodings": knownEncodings, "names": knownNames}
        
        with open(output_file, "wb") as f:
            f.write(pickle.dumps(data))
        
        logger.info(f"Training complete. Encodings saved to '{output_file}'")
        return True
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        return False

def main():
    """Main function to run model training."""
    success = train_model()
    if success:
        print("Model training completed successfully!")
    else:
        print("Model training failed!")
    return success

if __name__ == "__main__":
    main()
