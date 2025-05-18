"""
Obstacle module for the Hand Gesture Car Control Game.
"""
import pygame
import random
import math

class Obstacle:
    """
    מייצג מכשול במשחק
    """
    
    def __init__(self, x, y, obstacle_type="rock", width=40, height=40):
        """
        יוצר מכשול חדש
        
        Args:
            x, y: מיקום המכשול
            obstacle_type: סוג המכשול ("rock", "tree", "cone", "puddle")
            width, height: גודל המכשול
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.type = obstacle_type
        self.rotation = 0
        
        # מאפיינים ויזואליים לפי סוג המכשול
        if obstacle_type == "rock":
            self.color = (100, 100, 100)  # אפור
            self.shape = "circle"
            # גודל אקראי לסלעים
            size_factor = random.uniform(0.7, 1.5)
            self.width = int(width * size_factor)
            self.height = int(height * size_factor)
            
        elif obstacle_type == "tree":
            self.color = (0, 100, 0)  # ירוק כהה
            self.trunk_color = (101, 67, 33)  # חום
            self.shape = "tree"
            self.width = width
            self.height = height * 2  # עצים גבוהים יותר
            
        elif obstacle_type == "cone":
            self.color = (255, 140, 0)  # כתום
            self.shape = "triangle"
            self.width = width // 2  # קונוסים צרים יותר
            
        elif obstacle_type == "puddle":
            self.color = (0, 0, 150, 150)  # כחול עם שקיפות
            self.shape = "ellipse"
            # שלוליות יותר רחבות מגבוהות
            self.width = width * 2
            self.height = height
            
        else:  # סוג ברירת מחדל
            self.color = (200, 200, 0)  # צהוב
            self.shape = "rect"
        
        # היקף ההתנגשות (פשוט עיגול)
        self.collision_radius = max(self.width, self.height) // 2
        
        # תנועה (עבור מכשולים דינמיים)
        self.is_dynamic = False
        self.speed = 0
        self.direction = 0
        
    def update(self, dt):
        """
        מעדכן את מצב המכשול
        
        Args:
            dt: זמן שעבר מהעדכון האחרון
        """
        if self.is_dynamic:
            # חישוב וקטור תנועה
            rad = math.radians(self.direction)
            dx = math.sin(rad) * self.speed * dt
            dy = -math.cos(rad) * self.speed * dt
            
            # עדכון מיקום
            self.x += dx
            self.y += dy
            
            # סיבוב (אם המכשול מסתובב)
            self.rotation += dt * 10  # סיבוב איטי
            
    def draw(self, screen, offset_x, offset_y):
        """
        מצייר את המכשול על המסך
        
        Args:
            screen: משטח pygame לציור
            offset_x, offset_y: היסט העולם (מיקום הרכב)
        """
        # חישוב מיקום המכשול על המסך לפי היסט העולם
        screen_x = self.x - offset_x
        screen_y = self.y - offset_y
        
        # בדיקה אם המכשול בכלל נראה על המסך
        screen_width, screen_height = screen.get_size()
        if (screen_x + self.width < 0 or screen_x - self.width > screen_width or
            screen_y + self.height < 0 or screen_y - self.height > screen_height):
            return  # המכשול מחוץ למסך, אין צורך לצייר
        
        if self.shape == "circle":
            # ציור סלע (עיגול)
            pygame.draw.circle(
                screen,
                self.color,
                (int(screen_x), int(screen_y)),
                self.width // 2
            )
            
            # הוספת קווי טקסטורה לסלע
            for i in range(3):
                angle = math.radians(random.randint(0, 360))
                length = random.randint(5, self.width // 3)
                start_x = screen_x + math.cos(angle) * (self.width // 4)
                start_y = screen_y + math.sin(angle) * (self.width // 4)
                end_x = start_x + math.cos(angle) * length
                end_y = start_y + math.sin(angle) * length
                
                pygame.draw.line(
                    screen,
                    (70, 70, 70),  # אפור כהה יותר
                    (int(start_x), int(start_y)),
                    (int(end_x), int(end_y)),
                    2
                )
                
        elif self.shape == "tree":
            # ציור גזע העץ
            trunk_width = self.width // 3
            trunk_height = self.height // 2
            trunk_rect = pygame.Rect(
                screen_x - trunk_width // 2,
                screen_y - trunk_height // 2 + self.height // 4,  # קצת למטה
                trunk_width,
                trunk_height
            )
            pygame.draw.rect(screen, self.trunk_color, trunk_rect)
            
            # ציור צמרת העץ (עיגול ירוק)
            pygame.draw.circle(
                screen,
                self.color,
                (int(screen_x), int(screen_y - self.height // 4)),
                self.width // 2 + 5
            )
            
        elif self.shape == "triangle":
            # ציור קונוס (משולש)
            points = [
                (screen_x, screen_y - self.height // 2),  # קודקוד עליון
                (screen_x - self.width // 2, screen_y + self.height // 2),  # פינה שמאלית תחתונה
                (screen_x + self.width // 2, screen_y + self.height // 2)   # פינה ימנית תחתונה
            ]
            pygame.draw.polygon(screen, self.color, points)
            
            # פס לבן במרכז הקונוס
            pygame.draw.polygon(
                screen, 
                (255, 255, 255),
                [
                    (screen_x, screen_y - self.height // 4),
                    (screen_x - self.width // 4, screen_y + self.height // 4),
                    (screen_x + self.width // 4, screen_y + self.height // 4)
                ]
            )
            
        elif self.shape == "ellipse":
            # ציור שלולית (אליפסה)
            ellipse_rect = pygame.Rect(
                screen_x - self.width // 2,
                screen_y - self.height // 2,
                self.width,
                self.height
            )
            
            # צור משטח שקוף עם אליפסה
            surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            pygame.draw.ellipse(surface, self.color, (0, 0, self.width, self.height))
            
            # הוספת "גלים" קטנים בשלולית
            for i in range(2):
                pygame.draw.ellipse(
                    surface,
                    (100, 100, 255, 100),  # כחול בהיר שקוף
                    (
                        self.width // 4 + i * 10,
                        self.height // 4 + i * 5,
                        self.width // 2 - i * 20,
                        self.height // 2 - i * 10
                    ),
                    1  # רק קו (לא מילוי)
                )
            
            # הצגת המשטח על המסך
            screen.blit(surface, ellipse_rect)
            
        else:  # מלבן כברירת מחדל
            rect = pygame.Rect(
                screen_x - self.width // 2,
                screen_y - self.height // 2,
                self.width,
                self.height
            )
            pygame.draw.rect(screen, self.color, rect)
            
    def check_collision(self, car):
        """
        בודק התנגשות עם מכונית
        
        Args:
            car: אובייקט מכונית לבדיקה
            
        Returns:
            Boolean המציין אם יש התנגשות
        """
        # חישוב מרחק בין מרכז המכונית למרכז המכשול
        distance = math.sqrt((car.x - self.x)**2 + (car.y - self.y)**2)
        
        # התנגשות מתרחשת אם המרחק קטן מסכום רדיוסי ההתנגשות
        car_radius = max(car.width, car.height) // 2
        return distance < (car_radius + self.collision_radius)


class ObstacleManager:
    """
    מנהל את כל המכשולים במשחק
    """
    
    def __init__(self):
        """
        מאתחל את מנהל המכשולים
        """
        self.obstacles = []
        self.obstacle_types = ["rock", "tree", "cone", "puddle"]
        
    def generate_obstacles(self, num_obstacles, world_width, world_height, car_pos_x, car_pos_y, min_distance=150):
        """
        יוצר מספר מכשולים אקראיים בעולם
        
        Args:
            num_obstacles: מספר המכשולים ליצירה
            world_width, world_height: מימדי העולם
            car_pos_x, car_pos_y: מיקום המכונית
            min_distance: מרחק מינימלי בין מכשולים למכונית
        """
        self.obstacles = []
        
        for _ in range(num_obstacles):
            # ניסיון למצוא מיקום מתאים למכשול
            for attempt in range(10):  # מקסימום 10 ניסיונות למכשול
                # בחירת מיקום אקראי
                x = random.randint(50, world_width - 50)
                y = random.randint(50, world_height - 50)
                
                # וידוא מרחק מינימלי מהמכונית
                distance_to_car = math.sqrt((x - car_pos_x)**2 + (y - car_pos_y)**2)
                
                # בדיקה שהמכשול לא קרוב מדי למכשולים אחרים
                too_close = False
                for obstacle in self.obstacles:
                    distance = math.sqrt((x - obstacle.x)**2 + (y - obstacle.y)**2)
                    if distance < min_distance // 2:
                        too_close = True
                        break
                
                if distance_to_car >= min_distance and not too_close:
                    # מצאנו מיקום מתאים, יצירת מכשול
                    obstacle_type = random.choice(self.obstacle_types)
                    size = random.randint(30, 60)
                    self.obstacles.append(Obstacle(x, y, obstacle_type, size, size))
                    break
        
        print(f"נוצרו {len(self.obstacles)} מכשולים")
    
    def update(self, dt):
        """
        מעדכן את כל המכשולים
        
        Args:
            dt: זמן שעבר מהעדכון האחרון
        """
        for obstacle in self.obstacles:
            obstacle.update(dt)
    
    def draw(self, screen, offset_x, offset_y):
        """
        מצייר את כל המכשולים
        
        Args:
            screen: משטח pygame לציור
            offset_x, offset_y: היסט העולם
        """
        for obstacle in self.obstacles:
            obstacle.draw(screen, offset_x, offset_y)
    
    def check_collisions(self, car):
        """
        בודק התנגשויות עם מכונית
        
        Args:
            car: אובייקט מכונית לבדיקה
            
        Returns:
            רשימת מכשולים שהתנגשו במכונית
        """
        collisions = []
        for obstacle in self.obstacles:
            if obstacle.check_collision(car):
                collisions.append(obstacle)
        
        return collisions
