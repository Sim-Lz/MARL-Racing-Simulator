# track.py
import pygame
import math

class Track:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.surface = pygame.Surface((width, height))
        self.surface.fill((0, 0, 0))  
        self.brush_radius = 30 #larghezza massima della pista (la metà sono px per lato) 
        
        self.start_position = None
        self.start_angle = 0.0
        self.perimeter = 0.0
        self.drawing_points = []
        self.checkpoints = []  

    def draw_step(self, screen):
        mouse_pos = pygame.mouse.get_pos()
        buttons = pygame.mouse.get_pressed()
        
        if buttons[0]:  
            clamped_x = max(0, min(mouse_pos[0], self.width - 1))
            clamped_y = max(0, min(mouse_pos[1], self.height - 1))
            safe_pos = (clamped_x, clamped_y)

            if self.start_position is None:
                self.start_position = safe_pos
                self.drawing_points.append(safe_pos)
            else:
                last_p = self.drawing_points[-1]
                dist = math.hypot(safe_pos[0] - last_p[0], safe_pos[1] - last_p[1])
                
                if dist > 5:  
                    self.perimeter += dist
                    self.drawing_points.append(safe_pos)
                    
                    if len(self.drawing_points) == 6:
                        p0 = self.drawing_points[0]
                        p5 = self.drawing_points[5]
                        self.start_angle = math.degrees(math.atan2(p5[1] - p0[1], p5[0] - p0[0]))

            pygame.draw.circle(self.surface, (255, 255, 255), safe_pos, self.brush_radius)
        
        screen.blit(self.surface, (0, 0))
        pygame.draw.circle(screen, (0, 255, 0), mouse_pos, self.brush_radius, 2)
        
        font = pygame.font.SysFont("Arial", 20)
        text = font.render(f"Disegna la pista. Frecce SU/GIU per Spessore: {self.brush_radius*2}px | Premi INVIO per confermare.", True, (255, 255, 255))
        screen.blit(text, (20, 20))

    def close_track(self):
        """Sana il gap lineare tra l'ultimo punto registrato e il punto di partenza iniziale."""
        if len(self.drawing_points) < 2:
            return
            
        p_last = self.drawing_points[-1]
        p_first = self.drawing_points[0]
        dist = math.hypot(p_first[0] - p_last[0], p_first[1] - p_last[1])
        
        if dist > 5:
            steps = int(dist / 5)
            for i in range(1, steps + 1):
                t = i / steps
                curr_x = int(p_last[0] + (p_first[0] - p_last[0]) * t)
                curr_y = int(p_last[1] + (p_first[1] - p_last[1]) * t)
                
                self.drawing_points.append((curr_x, curr_y))
                pygame.draw.circle(self.surface, (255, 255, 255), (curr_x, curr_y), self.brush_radius)
                
            self.perimeter += dist

    def generate_checkpoints(self):
        """Estrae 12 checkpoint equidistanti posizionati lungo la mezzeria del circuito."""
        self.checkpoints.clear()
        total_pts = len(self.drawing_points)
        if total_pts < 20:
            return
        
        step = max(1, total_pts // 12)
        for i in range(0, total_pts, step):
            self.checkpoints.append(self.drawing_points[i])

    def get_pixel_safety(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            color = self.surface.get_at((int(x), int(y)))
            return color[0] > 50  
        return False