import pygame

class Tile:
    def __init__(self, x, y, tile_width, tile_height):
        self.x = x
        self.y = y
        self.pos = (x, y)
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.abs_x = x * tile_width
        self.abs_y = y * tile_height
        
        # Color Palette - Elegant Wood / Slate tones
        self.color = 'light' if (x + y) % 2 == 0 else 'dark'
        # Light: Off-white/Cream | Dark: Deep Charcoal/Navy
        self.draw_color = (240, 235, 225) if self.color == 'light' else (45, 52, 54)
        
        # Soft Highlight (instead of neon green)
        self.highlight_color = (180, 210, 180) if self.color == 'light' else (180, 210, 180)
        
        self.occupying_piece = None
        self.highlight = False
        self.rect = pygame.Rect(self.abs_x, self.abs_y, self.tile_width, self.tile_height)

    def draw(self, display):
        # 1. Draw Tile Base
        color = self.highlight_color if self.highlight else self.draw_color
        pygame.draw.rect(display, color, self.rect)

        # 2. Add a subtle inner border for a "premium" feel
        border_color = (200, 195, 185) if self.color == 'light' else (35, 40, 42)
        pygame.draw.rect(display, border_color, self.rect, 1)

        # 3. Draw Piece with a Drop Shadow
        if self.occupying_piece is not None:
            centering_rect = self.occupying_piece.img.get_rect()
            centering_rect.center = self.rect.center
            
            # --- Aesthetics: Drop Shadow ---
            # Create a small offset for a 3D effect
            shadow_pos = (centering_rect.x + 3, centering_rect.y + 3)
            # Create a shadow surface (black with transparency)
            shadow_surf = pygame.Surface(centering_rect.size, pygame.SRCALPHA)
            pygame.draw.circle(shadow_surf, (0, 0, 0, 80), 
                               (centering_rect.width//2, centering_rect.height//2), 
                               centering_rect.width//2 - 2)
            display.blit(shadow_surf, shadow_pos)
            
            # Draw the actual piece
            display.blit(self.occupying_piece.img, centering_rect.topleft)