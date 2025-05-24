import numpy as np
import cv2
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path
import os
import tensorflow as tf
import json

from app.models.models import BodyMeasurements, MeasurementUnit, Gender

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeasurementEngine:
    """
    Core engine for extracting body measurements from images or video.
    
    Uses MediaPipe for pose estimation and custom models for measurement extraction.
    """
    
    def __init__(self, model_path: Optional[str] = None, unit: MeasurementUnit = MeasurementUnit.CM):
        """
        Initialize the measurement engine with required models.
        
        Args:
            model_path: Path to the trained measurement extraction model
            unit: Default measurement unit (cm or inches)
        """
        self.unit = unit
        
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        
        # Initialize MediaPipe body segmentation
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        # Load measurement extraction model if provided
        self.measurement_model = None
        if model_path:
            self._load_model(model_path)
        
        # Reference object dimensions (defaults to credit card)
        self.reference_dimensions = {
            "credit_card": {"width": 8.56, "height": 5.398},  # cm
            "a4_paper": {"width": 21.0, "height": 29.7},  # cm
        }
    
    def _load_model(self, model_path: str):
        """
        Load the trained measurement model.
        
        Args:
            model_path: Path to the model file
        """
        try:
            self.measurement_model = tf.saved_model.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def process_image(self, image_path: str, height: float, 
                     gender: Optional[Gender] = None,
                     reference_object: Optional[Dict] = None) -> Tuple[BodyMeasurements, float]:
        """
        Extract measurements from a single image.
        
        Args:
            image_path: Path to the image file
            height: User's height in cm
            gender: User's gender (for model selection)
            reference_object: Reference object info for calibration
            
        Returns:
            Tuple of (BodyMeasurements, confidence_score)
        """
        # This is where we'll implement the image processing pipeline
        # Placeholder implementation
        logger.info(f"Processing image: {image_path}")
        
        # 1. Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Extract pose landmarks
        pose_landmarks = self._extract_pose_landmarks(image)
        
        # 3. Segment body from background
        segmentation_mask = self._segment_body(image)
        
        # 4. Calculate pixel-to-cm ratio using reference object or height
        scale_factor = self._calculate_scale_factor(
            pose_landmarks, 
            height, 
            reference_object
        )
        
        # 5. Extract measurements based on landmarks and segmentation
        raw_measurements = self._extract_measurements(
            image, 
            pose_landmarks, 
            segmentation_mask, 
            scale_factor,
            gender
        )
        
        # 6. Create and return BodyMeasurements object
        confidence_score = self._calculate_confidence_score(pose_landmarks, segmentation_mask)
        measurements = BodyMeasurements(
            height=height,
            chest=raw_measurements["chest"],
            waist=raw_measurements["waist"],
            hips=raw_measurements["hips"],
            inseam=raw_measurements.get("inseam"),
            shoulder_width=raw_measurements.get("shoulder_width"),
            arm_length=raw_measurements.get("arm_length"),
            unit=self.unit,
            confidence_score=confidence_score
        )
        
        return measurements, confidence_score
    
    def process_video(self, video_path: str, height: float,
                     gender: Optional[Gender] = None,
                     reference_object: Optional[Dict] = None) -> Tuple[BodyMeasurements, float]:
        """
        Extract measurements from video frames.
        
        Args:
            video_path: Path to the video file
            height: User's height in cm
            gender: User's gender (for model selection)
            reference_object: Reference object info for calibration
            
        Returns:
            Tuple of (BodyMeasurements, confidence_score)
        """
        # Placeholder for video processing implementation
        logger.info(f"Processing video: {video_path}")
        
        # Will implement frame extraction, pose estimation across frames,
        # and measurement aggregation from multiple frames
        
        # For now, placeholder implementation using first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read video from {video_path}")
        
        # Save frame temporarily and process as image
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, frame)
        measurements, confidence = self.process_image(
            temp_path, height, gender, reference_object
        )
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return measurements, confidence
    
    def _extract_pose_landmarks(self, image: np.ndarray) -> Optional[mp.solutions.pose.PoseLandmark]:
        """
        Extract pose landmarks from image using MediaPipe with enhanced validation.
        
        Args:
            image: Input image in RGB format
            
        Returns:
            Pose landmarks if valid pose is detected, None otherwise
            
        Raises:
            ValueError: If image is invalid or pose detection fails
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
            
        # Process image with MediaPipe
        results = self.pose.process(image)
        
        if not results.pose_landmarks:
            logger.warning("No pose landmarks detected in image")
            return None
            
        # Validate landmark visibility
        required_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        missing_landmarks = []
        for landmark in required_landmarks:
            if not results.pose_landmarks.landmark[landmark].visibility > 0.5:
                missing_landmarks.append(landmark.name)
                
        if missing_landmarks:
            logger.warning(f"Missing or low visibility landmarks: {', '.join(missing_landmarks)}")
            return None
            
        # Check pose orientation
        left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Calculate shoulder angle
        shoulder_angle = np.arctan2(
            right_shoulder.y - left_shoulder.y,
            right_shoulder.x - left_shoulder.x
        ) * 180 / np.pi
        
        # Check if person is facing forward (shoulders roughly horizontal)
        if abs(shoulder_angle) > 30:
            logger.warning(f"Person not facing forward (shoulder angle: {shoulder_angle:.1f} degrees)")
            return None
            
        # Check if person is standing upright
        left_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        hip_angle = np.arctan2(
            right_hip.y - left_hip.y,
            right_hip.x - left_hip.x
        ) * 180 / np.pi
        
        if abs(hip_angle) > 20:
            logger.warning(f"Person not standing upright (hip angle: {hip_angle:.1f} degrees)")
            return None
            
        # Calculate confidence score based on landmark visibility
        visibility_scores = [
            landmark.visibility 
            for landmark in results.pose_landmarks.landmark
            if landmark.visibility > 0
        ]
        
        if not visibility_scores:
            logger.warning("No visible landmarks found")
            return None
            
        avg_visibility = sum(visibility_scores) / len(visibility_scores)
        if avg_visibility < 0.7:
            logger.warning(f"Low average landmark visibility: {avg_visibility:.2f}")
            return None
            
        return results.pose_landmarks
    
    def _segment_body(self, image: np.ndarray) -> np.ndarray:
        """
        Segment body from background using MediaPipe with enhanced processing.
        
        Args:
            image: Input image in RGB format
            
        Returns:
            Binary mask of the segmented body
            
        Raises:
            ValueError: If segmentation fails or quality is too low
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
            
        # Get initial segmentation from MediaPipe
        results = self.segmentation.process(image)
        if not results.segmentation_mask is not None:
            raise ValueError("Segmentation failed - no mask generated")
            
        # Convert mask to binary
        mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            raise ValueError("No body contours detected")
            
        # Find the largest contour (should be the body)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a new mask from the largest contour
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)
        
        # Smooth the contour
        epsilon = 0.001 * cv2.arcLength(largest_contour, True)
        smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Create final mask with smoothed contour
        final_mask = np.zeros_like(mask)
        cv2.drawContours(final_mask, [smoothed_contour], -1, 255, -1)
        
        # Quality checks
        body_area = cv2.contourArea(largest_contour)
        image_area = image.shape[0] * image.shape[1]
        body_ratio = body_area / image_area
        
        if body_ratio < 0.1:  # Body too small
            raise ValueError(f"Body too small in image (ratio: {body_ratio:.2f})")
        elif body_ratio > 0.9:  # Body too large
            raise ValueError(f"Body too large in image (ratio: {body_ratio:.2f})")
            
        # Check for reasonable body proportions
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = h / w
        
        if aspect_ratio < 1.5 or aspect_ratio > 3.5:  # Unreasonable height/width ratio
            raise ValueError(f"Unreasonable body proportions (aspect ratio: {aspect_ratio:.2f})")
            
        # Apply final smoothing
        final_mask = cv2.GaussianBlur(final_mask, (5, 5), 0)
        final_mask = (final_mask > 127).astype(np.uint8) * 255
        
        return final_mask
    
    def _calculate_scale_factor(self, 
                               pose_landmarks: mp.solutions.pose.PoseLandmark,
                               height: float,
                               reference_object: Optional[Dict] = None) -> float:
        """
        Calculate the pixel-to-cm conversion factor using reference objects or height.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            height: User's height in cm
            reference_object: Dictionary containing reference object information
            
        Returns:
            float: Pixel-to-cm conversion factor
            
        Raises:
            ValueError: If scale factor calculation fails
        """
        if reference_object:
            try:
                # Get reference object type and dimensions
                ref_type = reference_object.get("type", "credit_card")
                known_dimensions = self.reference_dimensions.get(ref_type)
                
                if not known_dimensions:
                    raise ValueError(f"Unknown reference object type: {ref_type}")
                
                # Convert image to grayscale for processing
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                # Apply adaptive thresholding to detect objects
                thresh = cv2.adaptiveThreshold(
                    image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                
                # Find contours
                contours, _ = cv2.findContours(
                    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                
                if not contours:
                    raise ValueError("No objects detected in image")
                
                # Filter contours by area and aspect ratio
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 1000:  # Minimum area threshold
                        continue
                        
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Check if aspect ratio matches reference object
                    ref_ratio = known_dimensions["width"] / known_dimensions["height"]
                    if 0.8 * ref_ratio <= aspect_ratio <= 1.2 * ref_ratio:
                        valid_contours.append((contour, w))
                
                if not valid_contours:
                    raise ValueError("No matching reference object found")
                
                # Use the largest matching contour
                largest_contour, pixel_width = max(valid_contours, key=lambda x: x[1])
                
                # Calculate scale factor
                scale_factor = known_dimensions["width"] / pixel_width
                
                # Validate scale factor
                if scale_factor < 0.01 or scale_factor > 1.0:
                    raise ValueError(f"Invalid scale factor calculated: {scale_factor}")
                
                logger.info(f"Scale factor calculated using {ref_type}: {scale_factor:.4f} cm/pixel")
                return scale_factor
                
            except Exception as e:
                logger.warning(f"Reference object detection failed: {str(e)}")
                logger.info("Falling back to height-based scaling")
        
        # Fallback to height-based scaling
        if not pose_landmarks:
            raise ValueError("No pose landmarks available for height-based scaling")
        
        # Get key points for height calculation
        try:
            # Get ankle and head positions
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            
            # Use average of both ankles
            ankle_y = (left_ankle.y + right_ankle.y) / 2
            head_y = nose.y
            
            # Calculate pixel height
            pixel_height = abs(ankle_y - head_y) * image.shape[0]
            
            if pixel_height < 100:  # Minimum height threshold
                raise ValueError("Detected height too small")
            
            # Calculate scale factor
            scale_factor = height / pixel_height
            
            # Validate scale factor
            if scale_factor < 0.01 or scale_factor > 1.0:
                raise ValueError(f"Invalid scale factor calculated: {scale_factor}")
            
            logger.info(f"Scale factor calculated using height: {scale_factor:.4f} cm/pixel")
            return scale_factor
            
        except Exception as e:
            raise ValueError(f"Height-based scaling failed: {str(e)}")
    
    def _extract_measurements(self, 
                             image: np.ndarray,
                             pose_landmarks: mp.solutions.pose.PoseLandmark,
                             segmentation_mask: np.ndarray,
                             scale_factor: float,
                             gender: Optional[Gender] = None) -> Dict[str, float]:
        """
        Extract key body measurements based on pose landmarks and segmentation.
        
        Args:
            image: Input image
            pose_landmarks: MediaPipe pose landmarks
            segmentation_mask: Body segmentation mask
            scale_factor: Pixel-to-cm conversion factor
            gender: User's gender for model adjustment
            
        Returns:
            Dictionary of measurements in centimeters
        """
        try:
            # Get key landmarks for measurement points
            landmarks = {
                'left_shoulder': pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                'right_shoulder': pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                'left_hip': pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP],
                'right_hip': pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP],
                'left_knee': pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_KNEE],
                'right_knee': pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_KNEE],
                'left_ankle': pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE],
                'right_ankle': pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE],
                'left_wrist': pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST],
                'right_wrist': pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST],
                'left_elbow': pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                'right_elbow': pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            }
            
            # Convert landmarks to pixel coordinates
            height, width = image.shape[:2]
            pixel_landmarks = {
                name: (int(lm.x * width), int(lm.y * height))
                for name, lm in landmarks.items()
            }
            
            # Find contours in the segmentation mask
            contours, _ = cv2.findContours(
                segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                raise ValueError("No body contours found in segmentation mask")
            
            # Get the largest contour (body)
            body_contour = max(contours, key=cv2.contourArea)
            
            # Calculate shoulder width
            shoulder_width = self._calculate_width(
                body_contour,
                pixel_landmarks['left_shoulder'][1],
                pixel_landmarks['right_shoulder'][1],
                scale_factor
            )
            
            # Calculate chest width (at 20% down from shoulders)
            chest_y = pixel_landmarks['left_shoulder'][1] + int(
                (pixel_landmarks['left_hip'][1] - pixel_landmarks['left_shoulder'][1]) * 0.2
            )
            chest_width = self._calculate_width(
                body_contour,
                chest_y,
                chest_y,
                scale_factor
            )
            
            # Calculate waist width (at 50% down from shoulders)
            waist_y = pixel_landmarks['left_shoulder'][1] + int(
                (pixel_landmarks['left_hip'][1] - pixel_landmarks['left_shoulder'][1]) * 0.5
            )
            waist_width = self._calculate_width(
                body_contour,
                waist_y,
                waist_y,
                scale_factor
            )
            
            # Calculate hip width (at hip level)
            hip_width = self._calculate_width(
                body_contour,
                pixel_landmarks['left_hip'][1],
                pixel_landmarks['right_hip'][1],
                scale_factor
            )
            
            # Calculate arm length
            left_arm_length = self._calculate_length(
                pixel_landmarks['left_shoulder'],
                pixel_landmarks['left_elbow'],
                pixel_landmarks['left_wrist'],
                scale_factor
            )
            right_arm_length = self._calculate_length(
                pixel_landmarks['right_shoulder'],
                pixel_landmarks['right_elbow'],
                pixel_landmarks['right_wrist'],
                scale_factor
            )
            arm_length = (left_arm_length + right_arm_length) / 2
            
            # Calculate inseam
            left_inseam = self._calculate_length(
                pixel_landmarks['left_hip'],
                pixel_landmarks['left_knee'],
                pixel_landmarks['left_ankle'],
                scale_factor
            )
            right_inseam = self._calculate_length(
                pixel_landmarks['right_hip'],
                pixel_landmarks['right_knee'],
                pixel_landmarks['right_ankle'],
                scale_factor
            )
            inseam = (left_inseam + right_inseam) / 2
            
            # Apply gender-specific adjustments if needed
            if gender:
                measurements = self._apply_gender_adjustments(
                    {
                        'chest': chest_width,
                        'waist': waist_width,
                        'hips': hip_width,
                        'shoulder_width': shoulder_width,
                        'arm_length': arm_length,
                        'inseam': inseam
                    },
                    gender
                )
            else:
                measurements = {
                    'chest': chest_width,
                    'waist': waist_width,
                    'hips': hip_width,
                    'shoulder_width': shoulder_width,
                    'arm_length': arm_length,
                    'inseam': inseam
                }
            
            # Round all measurements to 1 decimal place
            return {k: round(v, 1) for k, v in measurements.items()}
            
        except Exception as e:
            logger.error(f"Error extracting measurements: {str(e)}")
            raise
    
    def _calculate_width(self, 
                        contour: np.ndarray,
                        y1: int,
                        y2: int,
                        scale_factor: float) -> float:
        """
        Calculate width at specific y-coordinates.
        
        Args:
            contour: Body contour
            y1, y2: Y-coordinates to measure width at
            scale_factor: Pixel-to-cm conversion factor
            
        Returns:
            Width in centimeters
        """
        # Create horizontal lines at specified y-coordinates
        line1 = np.array([[0, y1], [contour.shape[1], y1]])
        line2 = np.array([[0, y2], [contour.shape[1], y2]])
        
        # Find intersections with contour
        intersections1 = self._find_contour_intersections(contour, line1)
        intersections2 = self._find_contour_intersections(contour, line2)
        
        if not intersections1 or not intersections2:
            raise ValueError("Could not find contour intersections")
        
        # Calculate widths
        width1 = abs(intersections1[1][0] - intersections1[0][0])
        width2 = abs(intersections2[1][0] - intersections2[0][0])
        
        # Return average width in centimeters
        return (width1 + width2) / 2 * scale_factor
    
    def _calculate_length(self,
                         point1: Tuple[int, int],
                         point2: Tuple[int, int],
                         point3: Tuple[int, int],
                         scale_factor: float) -> float:
        """
        Calculate length along a path of three points.
        
        Args:
            point1, point2, point3: Points along the path
            scale_factor: Pixel-to-cm conversion factor
            
        Returns:
            Length in centimeters
        """
        # Calculate distances between points
        dist1 = np.sqrt(
            (point2[0] - point1[0])**2 + 
            (point2[1] - point1[1])**2
        )
        dist2 = np.sqrt(
            (point3[0] - point2[0])**2 + 
            (point3[1] - point2[1])**2
        )
        
        # Return total length in centimeters
        return (dist1 + dist2) * scale_factor
    
    def _find_contour_intersections(self,
                                  contour: np.ndarray,
                                  line: np.ndarray) -> List[Tuple[int, int]]:
        """
        Find intersections between a contour and a line.
        
        Args:
            contour: Body contour
            line: Line to find intersections with
            
        Returns:
            List of intersection points
        """
        # Create a blank image
        img = np.zeros((contour.shape[0], contour.shape[1]), dtype=np.uint8)
        
        # Draw the contour and line
        cv2.drawContours(img, [contour], -1, 255, 1)
        cv2.line(img, tuple(line[0]), tuple(line[1]), 255, 1)
        
        # Find intersections
        intersections = []
        for i in range(len(contour)):
            pt1 = tuple(contour[i][0])
            pt2 = tuple(contour[(i + 1) % len(contour)][0])
            
            # Check if line segment intersects with the horizontal line
            if (pt1[1] <= line[0][1] and pt2[1] >= line[0][1]) or \
               (pt1[1] >= line[0][1] and pt2[1] <= line[0][1]):
                # Calculate intersection point
                x = int(pt1[0] + (pt2[0] - pt1[0]) * (line[0][1] - pt1[1]) / (pt2[1] - pt1[1]))
                intersections.append((x, int(line[0][1])))
        
        # Sort intersections by x-coordinate
        intersections.sort(key=lambda p: p[0])
        
        return intersections
    
    def _apply_gender_adjustments(self,
                                measurements: Dict[str, float],
                                gender: Gender) -> Dict[str, float]:
        """
        Apply gender-specific adjustments to measurements.
        
        Args:
            measurements: Raw measurements
            gender: User's gender
            
        Returns:
            Adjusted measurements
        """
        # Define adjustment factors based on gender
        adjustments = {
            Gender.MALE: {
                'chest': 1.05,  # Slightly larger chest
                'waist': 0.95,  # Slightly smaller waist
                'hips': 0.90,   # Smaller hips
                'shoulder_width': 1.10  # Wider shoulders
            },
            Gender.FEMALE: {
                'chest': 0.95,  # Slightly smaller chest
                'waist': 1.05,  # Slightly larger waist
                'hips': 1.10,   # Larger hips
                'shoulder_width': 0.90  # Narrower shoulders
            }
        }
        
        # Get adjustment factors for the gender
        factors = adjustments.get(gender, {})
        
        # Apply adjustments
        adjusted = measurements.copy()
        for measurement, factor in factors.items():
            if measurement in adjusted:
                adjusted[measurement] *= factor
        
        return adjusted
    
    def _calculate_confidence_score(self, 
                                  pose_landmarks: mp.solutions.pose.PoseLandmark,
                                  segmentation_mask: np.ndarray) -> float:
        """
        Calculate confidence score for the measurements based on multiple factors.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            segmentation_mask: Body segmentation mask
            
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            confidence_factors = []
            
            # 1. Pose Landmark Visibility Score
            landmark_visibility = self._calculate_landmark_visibility(pose_landmarks)
            confidence_factors.append(('landmark_visibility', landmark_visibility))
            
            # 2. Segmentation Quality Score
            segmentation_quality = self._assess_segmentation_quality(segmentation_mask)
            confidence_factors.append(('segmentation_quality', segmentation_quality))
            
            # 3. Pose Stability Score
            pose_stability = self._assess_pose_stability(pose_landmarks)
            confidence_factors.append(('pose_stability', pose_stability))
            
            # 4. Body Proportion Score
            body_proportions = self._assess_body_proportions(pose_landmarks)
            confidence_factors.append(('body_proportions', body_proportions))
            
            # Calculate weighted average of all factors
            weights = {
                'landmark_visibility': 0.4,
                'segmentation_quality': 0.3,
                'pose_stability': 0.2,
                'body_proportions': 0.1
            }
            
            final_score = sum(
                score * weights[factor]
                for factor, score in confidence_factors
            )
            
            # Log confidence factors for debugging
            logger.debug("Confidence factors:")
            for factor, score in confidence_factors:
                logger.debug(f"{factor}: {score:.2f}")
            logger.debug(f"Final confidence score: {final_score:.2f}")
            
            return round(final_score, 2)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.0
    
    def _calculate_landmark_visibility(self, 
                                    pose_landmarks: mp.solutions.pose.PoseLandmark) -> float:
        """
        Calculate visibility score for required landmarks.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            
        Returns:
            float: Visibility score between 0 and 1
        """
        required_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        ]
        
        visibility_scores = []
        for landmark in required_landmarks:
            visibility = pose_landmarks.landmark[landmark].visibility
            visibility_scores.append(visibility)
        
        return sum(visibility_scores) / len(visibility_scores)
    
    def _assess_segmentation_quality(self, segmentation_mask: np.ndarray) -> float:
        """
        Assess the quality of body segmentation.
        
        Args:
            segmentation_mask: Binary mask of segmented body
            
        Returns:
            float: Quality score between 0 and 1
        """
        try:
            # Find contours
            contours, _ = cv2.findContours(
                segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if not contours:
                return 0.0
            
            # Get the largest contour
            body_contour = max(contours, key=cv2.contourArea)
            
            # Calculate contour area
            area = cv2.contourArea(body_contour)
            total_area = segmentation_mask.shape[0] * segmentation_mask.shape[1]
            
            # Check if body area is reasonable (between 10% and 90% of image)
            if area < 0.1 * total_area or area > 0.9 * total_area:
                return 0.3
            
            # Calculate contour smoothness
            perimeter = cv2.arcLength(body_contour, True)
            smoothness = 4 * np.pi * area / (perimeter * perimeter)
            
            # Calculate contour complexity
            complexity = len(body_contour) / perimeter
            
            # Combine factors
            area_score = 1.0 if 0.2 <= area/total_area <= 0.8 else 0.5
            smoothness_score = min(smoothness * 2, 1.0)
            complexity_score = min(complexity / 0.1, 1.0)
            
            return (area_score * 0.4 + smoothness_score * 0.3 + complexity_score * 0.3)
            
        except Exception as e:
            logger.error(f"Error assessing segmentation quality: {str(e)}")
            return 0.0
    
    def _assess_pose_stability(self, 
                             pose_landmarks: mp.solutions.pose.PoseLandmark) -> float:
        """
        Assess the stability of the detected pose.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            
        Returns:
            float: Stability score between 0 and 1
        """
        try:
            # Check shoulder alignment
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            shoulder_angle = abs(np.arctan2(
                right_shoulder.y - left_shoulder.y,
                right_shoulder.x - left_shoulder.x
            ) * 180 / np.pi)
            
            # Check hip alignment
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            hip_angle = abs(np.arctan2(
                right_hip.y - left_hip.y,
                right_hip.x - left_hip.x
            ) * 180 / np.pi)
            
            # Calculate scores
            shoulder_score = max(0, 1 - shoulder_angle / 30)
            hip_score = max(0, 1 - hip_angle / 20)
            
            return (shoulder_score * 0.6 + hip_score * 0.4)
            
        except Exception as e:
            logger.error(f"Error assessing pose stability: {str(e)}")
            return 0.0
    
    def _assess_body_proportions(self, 
                               pose_landmarks: mp.solutions.pose.PoseLandmark) -> float:
        """
        Assess if body proportions are within reasonable ranges.
        
        Args:
            pose_landmarks: MediaPipe pose landmarks
            
        Returns:
            float: Proportion score between 0 and 1
        """
        try:
            # Get key points
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_ankle = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
            
            # Calculate proportions
            shoulder_width = np.sqrt(
                (right_shoulder.x - left_shoulder.x)**2 +
                (right_shoulder.y - left_shoulder.y)**2
            )
            
            hip_width = np.sqrt(
                (right_hip.x - left_hip.x)**2 +
                (right_hip.y - left_hip.y)**2
            )
            
            torso_height = abs(left_shoulder.y - left_hip.y)
            leg_height = abs(left_hip.y - left_ankle.y)
            
            # Calculate ratios
            shoulder_hip_ratio = shoulder_width / hip_width
            torso_leg_ratio = torso_height / leg_height
            
            # Score the ratios
            shoulder_hip_score = 1.0 if 0.8 <= shoulder_hip_ratio <= 1.2 else 0.5
            torso_leg_score = 1.0 if 0.8 <= torso_leg_ratio <= 1.2 else 0.5
            
            return (shoulder_hip_score * 0.5 + torso_leg_score * 0.5)
            
        except Exception as e:
            logger.error(f"Error assessing body proportions: {str(e)}")
            return 0.0
    
    def convert_units(self, measurements: BodyMeasurements, 
                     target_unit: MeasurementUnit) -> BodyMeasurements:
        """
        Convert measurements between cm and inches.
        
        Args:
            measurements: Original measurements
            target_unit: Target unit for conversion
            
        Returns:
            Converted measurements
        """
        if measurements.unit == target_unit:
            return measurements
        
        conversion_factor = 0.393701 if target_unit == MeasurementUnit.INCHES else 2.54
        
        # Create a new measurements object with converted values
        converted = BodyMeasurements(
            height=measurements.height * conversion_factor,
            weight=measurements.weight,  # Weight is not converted (would need separate logic)
            chest=measurements.chest * conversion_factor,
            waist=measurements.waist * conversion_factor,
            hips=measurements.hips * conversion_factor,
            unit=target_unit,
            confidence_score=measurements.confidence_score
        )
        
        # Convert optional measurements if present
        if measurements.inseam:
            converted.inseam = measurements.inseam * conversion_factor
        if measurements.shoulder_width:
            converted.shoulder_width = measurements.shoulder_width * conversion_factor
        if measurements.arm_length:
            converted.arm_length = measurements.arm_length * conversion_factor
            
        return converted 