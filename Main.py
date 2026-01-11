import pygame
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from checkers.Board import Board
from checkers.Game import Game
from HomographyAlgorithm.homography import CheckerBoardDetector

YOLO_MODEL_PATH = "YoloPreTrained/resultados/runs_3_classes/detect/checkers_yolo12/weights/best.pt"
WINDOW_SIZE = (1000, 1000)

pygame.init()

class CheckersApp:
    def __init__(self, screen):
        self.screen = screen
        self.running = True
        self.FPS = pygame.time.Clock()
        self.state = "MENU" 
        
        self.detector = CheckerBoardDetector(YOLO_MODEL_PATH, board_size=8)
        
        # Image Adjustment Variables
        self.raw_image = None       # The full res CV2 image
        self.display_image = None   # The scaled pygame surface
        self.corners = []           # Corners in RAW image coordinates
        self.scale_factor = 1.0     # Ratio between display and raw
        self.img_offset = (0, 0)    # (x, y) offset to center image in window
        self.selected_corner_idx = None
        
        # Game logic
        self.board = None
        self.game = None
        self.font = pygame.font.SysFont("Arial", 24)

    def convert_matrix_to_board_config(self, matrix):
        """
        Converts Homography integer matrix to Board.py string config.
        Assumption: 
        0 = Empty
        1 = Black Piece (YOLO ID 0) -> 'bp'
        2 = White Piece (YOLO ID 1) -> 'rp' (Assuming White in CV is Red in PyGame)
        """
        config = []
        for row in matrix:
            new_row = []
            for cell in row:
                if cell == 0:
                    new_row.append('')
                elif cell == 2:
                    new_row.append('bp')
                elif cell == 3:
                    new_row.append('rp') 
            config.append(new_row)
        return config

    def open_file_dialog(self):
        """Opens native file explorer to choose image."""
        root = tk.Tk()
        root.withdraw() # Hide the main window
        file_path = filedialog.askopenfilename(
            title="Select Board Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg")]
        )
        root.destroy()
        return file_path

    def load_and_detect(self):
        """Loads image, runs detection, prepares adjustment screen."""
        path = self.open_file_dialog()
        if not path:
            return # User cancelled

        print(f"Loading {path}...")
        self.raw_image = cv2.imread(path)
        
        # Detect initial corners
        print("Detecting corners...")
        self.corners = self.detector.detect_board_corners(self.raw_image)
        
        # Calculate scaling to fit PyGame window
        img_h, img_w = self.raw_image.shape[:2]
        win_w, win_h = WINDOW_SIZE
        
        scale_w = win_w / img_w
        scale_h = win_h / img_h
        self.scale_factor = min(scale_w, scale_h) * 0.9 # 90% fit
        
        new_w = int(img_w * self.scale_factor)
        new_h = int(img_h * self.scale_factor)
        
        # Create display surface
        rgb_img = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_img, (new_w, new_h))
        self.display_image = pygame.image.frombuffer(resized.tobytes(), resized.shape[1::-1], "RGB")
        
        # Center offset
        self.img_offset = ((win_w - new_w)//2, (win_h - new_h)//2)
        
        self.state = "ADJUST_CORNERS"

    def process_and_start_game(self):
        """Finalize corners, run Homography/YOLO, start Game."""
        print("Processing board...")
        
        # 1. Update detector with manual corners
        self.detector.set_corners(self.corners)
        
        # 2. Get Matrix (Heavy processing)
        # Note: You might want to show a "Loading" screen here
        matrix = self.detector.get_matrix_from_image(self.raw_image, scale_factor=8)
        print("Matrix extracted:\n", matrix)
        
        # 3. Convert to Board Config
        board_config = self.convert_matrix_to_board_config(matrix)
        
        # 4. Init Game
        tile_width = WINDOW_SIZE[0] // 8
        tile_height = WINDOW_SIZE[1] // 8
        self.board = Board(tile_width, tile_height, 8, custom_config=board_config)
        self.game = Game()
        
        self.state = "PLAYING"

    # --- INPUT HANDLING ---

    def handle_adjust_input(self, event):
        """Handle mouse dragging for corners."""
        # Convert mouse pos to raw image coordinates
        mx, my = pygame.mouse.get_pos()
        ox, oy = self.img_offset
        
        # Map mouse back to raw image space
        raw_mx = (mx - ox) / self.scale_factor
        raw_my = (my - oy) / self.scale_factor
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Check if clicked near a corner (threshold 30px scaled)
            threshold = 30 / self.scale_factor
            distances = [np.linalg.norm(np.array([raw_mx, raw_my]) - c) for c in self.corners]
            min_dist = min(distances)
            if min_dist < threshold:
                self.selected_corner_idx = np.argmin(distances)
                
        elif event.type == pygame.MOUSEBUTTONUP:
            self.selected_corner_idx = None
            
        elif event.type == pygame.MOUSEMOTION:
            if self.selected_corner_idx is not None:
                self.corners[self.selected_corner_idx] = [raw_mx, raw_my]
                
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.process_and_start_game()

    # --- DRAWING ---

    def draw_menu(self):
        self.screen.fill((30, 30, 30))
        text = self.font.render("Press SPACE to Upload Board Image", True, (255, 255, 255))
        rect = text.get_rect(center=(WINDOW_SIZE[0]//2, WINDOW_SIZE[1]//2))
        self.screen.blit(text, rect)

    def draw_adjust(self):
        self.screen.fill((20, 20, 20))
        
        # Draw Image
        if self.display_image:
            self.screen.blit(self.display_image, self.img_offset)
        
        # Draw Corners and Lines
        ox, oy = self.img_offset
        
        # Convert raw corners to screen coords for drawing
        screen_corners = []
        for c in self.corners:
            sx = int(c[0] * self.scale_factor + ox)
            sy = int(c[1] * self.scale_factor + oy)
            screen_corners.append((sx, sy))
            
        # Draw Polygon connecting corners
        if len(screen_corners) == 4:
            pygame.draw.lines(self.screen, (0, 255, 0), True, screen_corners, 2)
            
        # Draw Corner Points
        labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        for i, (sx, sy) in enumerate(screen_corners):
            color = (0, 255, 255) if i == self.selected_corner_idx else (255, 0, 0)
            pygame.draw.circle(self.screen, color, (sx, sy), 8)
            
            lbl = self.font.render(labels[i], True, (255, 255, 255))
            self.screen.blit(lbl, (sx + 10, sy - 10))

        # Instructions
        inst = self.font.render("Drag Corners. Press ENTER to Confirm.", True, (255, 255, 255))
        self.screen.blit(inst, (20, 20))

    def main_loop(self):
        while self.running:
            # Event Loop
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                if self.state == "MENU":
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                        self.load_and_detect()
                        
                elif self.state == "ADJUST_CORNERS":
                    self.handle_adjust_input(event)
                    
                elif self.state == "PLAYING":
                    if not self.game.is_game_over(self.board):
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            self.board.handle_click(event.pos)
                        # Helper reset
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                             self.state = "MENU" # Go back to menu
                    else:
                        self.game.message()
                        self.running = False

            # Draw Loop
            if self.state == "MENU":
                self.draw_menu()
            elif self.state == "ADJUST_CORNERS":
                self.draw_adjust()
            elif self.state == "PLAYING":
                self.board.draw(self.screen)
                self.game.check_jump(self.board)

            pygame.display.update()
            self.FPS.tick(60)

if __name__ == "__main__":
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Smart Checkers")
    
    app = CheckersApp(screen)
    app.main_loop()
    pygame.quit()