import cv2
import numpy as np
from ultralytics import YOLO

class CheckerBoardDetector:
    def __init__(self, yolo_model_path, board_size=8):
        self.board_size = board_size
        self.yolo_model = YOLO(yolo_model_path)
        self.board_corners = None
        self.homography_matrix = None
        self.warped_image = None

    def detect_board_corners(self, image):
        """Detects board corners using HoughLinesP method."""
        corners = self._detect_via_hough_linesP(image)
        
        # Last resort - use image boundaries with margin
        if corners is None:
            print("HoughLinesP detection failed. Using image boundaries as corners.")
            h, w = image.shape[:2]
            margin = 20
            corners = np.array([
                [margin, margin], 
                [w - margin, margin], 
                [w - margin, h - margin], 
                [margin, h - margin]
            ], dtype=np.float32)

        self.board_corners = self._order_corners(corners)
        return self.board_corners

    def _detect_via_hough_linesP(self, image):
        """
        Detects board using HoughLinesP to find grid line segments.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Use adaptive thresholding instead of Canny for better edge detection
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )
        
        # Get edges from threshold
        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        
        # HoughLinesP finds line segments
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=30, maxLineGap=15)
        
        if lines is None or len(lines) < 4:
            return None

        # Extract all points from line segments
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.append([x1, y1])
            points.append([x2, y2])

        if len(points) < 4:
            return None

        points = np.array(points, dtype=np.float32)
        
        # Convex Hull wraps the outermost points
        hull = cv2.convexHull(points)
        
        # Approximate hull to a quadrilateral
        epsilon = 0.08 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        corners = approx.reshape(-1, 2).astype(np.float32)
        
        if len(corners) >= 4:
            hull = cv2.convexHull(corners)
            corners = hull.reshape(-1, 2).astype(np.float32)
        
        if len(corners) == 4:
            return corners
        
        return None

    def _order_corners(self, pts):
        """Orders corners consistently: TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype="float32")
        
        # Sum: top-left has smallest sum, bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # Top-left
        rect[2] = pts[np.argmax(s)]  # Bottom-right
        
        # Difference: top-right has smallest diff, bottom-left has largest
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # Top-right
        rect[3] = pts[np.argmax(diff)]  # Bottom-left
        
        return rect

    def set_corners(self, corners):
        """Manually set board corners from UI."""
        self.board_corners = np.array(corners, dtype=np.float32)
        print(self.board_corners)

    def get_matrix_from_image(self, image, scale_factor=9):
        """
        Warps the image via Homography Reconstruction to get a top-down view.
        Then runs YOLO to detect pieces and generate the board matrix.
        """
        if self.board_corners is None:
            raise ValueError("Board corners not set. Run detect_board_corners first.")

        # Create output size based on board dimensions
        output_size = int(self.board_size * 100 * scale_factor)
        
        # Define destination points for a perfectly aligned board
        dst_pts = np.array([
            [0, 0],
            [output_size, 0],
            [output_size, output_size],
            [0, output_size]
        ], dtype=np.float32)

        # Compute homography matrix to transform tilted board to top-down view
        self.homography_matrix, _ = cv2.findHomography(self.board_corners, dst_pts)
        
        # Warp perspective to get straight-on board view
        self.warped_image = cv2.warpPerspective(image, self.homography_matrix, (output_size, output_size))

        # Run YOLO on warped image
        results = self.yolo_model(self.warped_image, conf=0.25, verbose=False)
        
        detections = []
        if results and results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                class_id = int(box.cls[0].item())
                conf = box.conf[0].item()
                detections.append({'cls': class_id, 'conf': conf, 'cx': cx, 'cy': cy})

        # Build board matrix
        matrix = np.zeros((self.board_size, self.board_size), dtype=int)
        cell_size = output_size / self.board_size
        candidates = {}

        for det in detections:
            col = int(det['cx'] // cell_size)
            row = int(det['cy'] // cell_size)

            if 0 <= col < self.board_size and 0 <= row < self.board_size:
                if (row, col) not in candidates:
                    candidates[(row, col)] = []
                candidates[(row, col)].append(det)

        for (r, c), dets in candidates.items():
            best = max(dets, key=lambda x: x['conf'])
            matrix[r, c] = best['cls'] + 1

        return matrix