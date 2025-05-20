"""
Road Generator module for Hand Gesture Car Control Game.
Creates roads with obstacles as specified in the requirements.
"""

import random
import math
import pygame
from game.obstacle import Obstacle

class RoadSegment:
    """
    מייצג קטע דרך בודד במשחק
    """
    def __init__(self, start_point, end_point, width=300, segment_id=0):
        """
        מאתחל קטע דרך חדש
        
        Args:
            start_point: נקודת התחלה (x, y)
            end_point: נקודת סיום (x, y)
            width: רוחב הדרך
            segment_id: מזהה הקטע (לצורך מעקב)
        """
        self.start = start_point
        self.end = end_point
        self.width = width
        self.id = segment_id
        
        # חישוב כיוון הקטע
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        self.direction = math.degrees(math.atan2(dy, dx))
        
        # נקודות שוליים
        self.left_edge = []
        self.right_edge = []
        self._calculate_edges()
        
        # רשימת מכשולים בקטע זה
        self.obstacles = []
        
        # מידע נוסף
        self.difficulty = 1.0  # מקדם קושי לקטע (1.0 = רגיל)
        self.visited = False   # האם השחקן כבר עבר בקטע
        self.length = math.sqrt(dx*dx + dy*dy)  # אורך הקטע

    def _calculate_edges(self):
        """מחשב את נקודות השוליים של קטע הדרך"""
        # חישוב וקטור כיוון מנורמל
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            dx /= length
            dy /= length
        
        # חישוב וקטור ניצב
        perpx = -dy
        perpy = dx
        
        # חישוב נקודות שוליים
        half_width = self.width / 2
        
        # שולי התחלה
        left_start = (
            self.start[0] + perpx * half_width,
            self.start[1] + perpy * half_width
        )
        right_start = (
            self.start[0] - perpx * half_width,
            self.start[1] - perpy * half_width
        )
        
        # שולי סיום
        left_end = (
            self.end[0] + perpx * half_width,
            self.end[1] + perpy * half_width
        )
        right_end = (
            self.end[0] - perpx * half_width,
            self.end[1] - perpy * half_width
        )
        
        self.left_edge = [left_start, left_end]
        self.right_edge = [right_start, right_end]

    def generate_obstacles(self, obstacle_density=1.0, obstacle_types=None):
        """
        יוצר מכשולים לאורך קטע הדרך
        
        Args:
            obstacle_density: צפיפות מכשולים (1.0 = רגיל)
            obstacle_types: רשימת סוגי מכשולים אפשריים
        """
        if obstacle_types is None:
            obstacle_types = ["rock", "tree", "cone", "puddle"]
            
        # חישוב מספר המכשולים על פי אורך הקטע וצפיפות
        num_obstacles = int((self.length / 200) * obstacle_density)
        min_obstacles = max(1, int(obstacle_density))
        num_obstacles = max(min_obstacles, num_obstacles)
        
        # סידור המכשולים לאורך הקטע
        self.obstacles = []
        
        # יצירת מכשולים בצד שמאל של הדרך
        for _ in range(num_obstacles):
            # מיקום אקראי לאורך הקטע (0.0 עד 1.0)
            t = random.uniform(0.1, 0.9)
            
            # חישוב קואורדינטות הבסיס על הקו
            base_x = self.start[0] + (self.end[0] - self.start[0]) * t
            base_y = self.start[1] + (self.end[1] - self.start[1]) * t
            
            # הוספת מיקום אקראי בצד שמאל של הדרך
            dx = self.left_edge[0][0] - self.start[0]
            dy = self.left_edge[0][1] - self.start[1]
            # קצת מעבר לשוליים
            margin = random.uniform(30, 100)
            obstacle_x = base_x + dx + (dx * margin / self.width)
            obstacle_y = base_y + dy + (dy * margin / self.width)
            
            # יצירת מכשול אקראי
            obstacle_type = random.choice(obstacle_types)
            size = random.randint(30, 60)
            obstacle = Obstacle(obstacle_x, obstacle_y, obstacle_type, size, size)
            self.obstacles.append(obstacle)
        
        # יצירת מכשולים בצד ימין של הדרך
        for _ in range(num_obstacles):
            t = random.uniform(0.1, 0.9)
            base_x = self.start[0] + (self.end[0] - self.start[0]) * t
            base_y = self.start[1] + (self.end[1] - self.start[1]) * t
            
            # הוספת מיקום אקראי בצד ימין של הדרך
            dx = self.right_edge[0][0] - self.start[0]
            dy = self.right_edge[0][1] - self.start[1]
            margin = random.uniform(30, 100)
            obstacle_x = base_x + dx + (dx * margin / self.width)
            obstacle_y = base_y + dy + (dy * margin / self.width)
            
            # יצירת מכשול אקראי
            obstacle_type = random.choice(obstacle_types)
            size = random.randint(30, 60)
            obstacle = Obstacle(obstacle_x, obstacle_y, obstacle_type, size, size)
            self.obstacles.append(obstacle)
            

class RoadGenerator:
    """
    מנהל את יצירת הדרכים והמכשולים במשחק
    """
    def __init__(self, world_width, world_height, obstacle_manager=None):
        """
        מאתחל גנרטור דרכים חדש
        
        Args:
            world_width: רוחב עולם המשחק
            world_height: גובה עולם המשחק
            obstacle_manager: מנהל המכשולים הקיים (אופציונלי)
        """
        self.world_width = world_width
        self.world_height = world_height
        self.obstacle_manager = obstacle_manager
        
        # מאפייני הדרך
        self.road_width = 300
        self.segment_length = 500
        self.max_turn_angle = 45  # זווית הפניה המרבית בין קטעים
        
        # רשימת קטעי הדרך
        self.road_segments = []
        
        # נקודת סיום אחרונה (לשרשור קטעים)
        self.last_end_point = (world_width // 2, world_height // 2)
        self.last_direction = 0  # כיוון אחרון (צפון)
        
        # מונה קטעים
        self.segment_counter = 0
        
    def generate_road_segment(self, num_segments=1, direction_bias="north"):
        """
        יוצר קטעי דרך חדשים עם כיוון כללי צפונה
        
        Args:
            num_segments: מספר הקטעים ליצירה
            direction_bias: העדפת כיוון ("north", "random", etc.)
            
        Returns:
            רשימת קטעי הדרך החדשים
        """
        new_segments = []
        
        for _ in range(num_segments):
            # קביעת כיוון הקטע החדש
            if direction_bias == "north":
                # מקסימום 45 מעלות סטייה מצפון
                max_deviation = self.max_turn_angle
                # 0 מעלות = צפון, סטייה שמאלה או ימינה
                new_direction = random.uniform(-max_deviation, max_deviation)
                
                # אם יש קטע קודם, לוקחים בחשבון את הכיוון שלו
                if self.road_segments:
                    # מגבילים את השינוי בין הקטעים ל-45 מעלות
                    direction_change = random.uniform(-max_deviation/2, max_deviation/2)
                    new_direction = self.last_direction + direction_change
                    
                    # וידוא שהכיוון לא סוטה יותר מ-45 מעלות מצפון
                    if new_direction < -max_deviation:
                        new_direction = -max_deviation
                    elif new_direction > max_deviation:
                        new_direction = max_deviation
            else:
                # כיוון אקראי (למקרה שנרצה בעתיד)
                new_direction = random.uniform(0, 360)
            
            # חישוב נקודת הסיום על פי הכיוון והאורך
            rad = math.radians(new_direction)
            end_x = self.last_end_point[0] + math.sin(rad) * self.segment_length
            end_y = self.last_end_point[1] - math.cos(rad) * self.segment_length
            
            # וידוא שהקטע בתוך גבולות העולם
            if end_x < 0:
                end_x = 0
            elif end_x > self.world_width:
                end_x = self.world_width
                
            if end_y < 0:
                end_y = 0
            elif end_y > self.world_height:
                end_y = self.world_height
            
            # יצירת קטע דרך חדש
            new_segment = RoadSegment(
                self.last_end_point,
                (end_x, end_y),
                self.road_width,
                self.segment_counter
            )
            
            # הוספת מכשולים לקטע
            new_segment.generate_obstacles()
            
            # עדכון רשימת המכשולים הראשית
            if self.obstacle_manager is not None:
                for obstacle in new_segment.obstacles:
                    self.obstacle_manager.obstacles.append(obstacle)
            
            # הוספת הקטע לרשימה
            self.road_segments.append(new_segment)
            new_segments.append(new_segment)
            
            # עדכון הנקודה האחרונה והכיוון
            self.last_end_point = (end_x, end_y)
            self.last_direction = new_direction
            self.segment_counter += 1
        
        return new_segments
    
    def draw_road(self, screen, offset_x, offset_y):
        """
        מצייר את כל קטעי הדרך על המסך
        
        Args:
            screen: משטח הציור של pygame
            offset_x, offset_y: היסט העולם לצורך מרכוז המכונית
        """
        for segment in self.road_segments:
            # המרת נקודות מעולם המשחק למיקום על המסך
            start_x = segment.start[0] - offset_x
            start_y = segment.start[1] - offset_y
            end_x = segment.end[0] - offset_x
            end_y = segment.end[1] - offset_y
            
            # חישוב וקטור כיוון מנורמל
            dx = end_x - start_x
            dy = end_y - start_y
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx /= length
                dy /= length
            
            # חישוב וקטור ניצב
            perpx = -dy
            perpy = dx
            
            # חישוב ארבע הנקודות של המלבן
            half_width = segment.width / 2
            points = [
                (start_x + perpx * half_width, start_y + perpy * half_width),
                (end_x + perpx * half_width, end_y + perpy * half_width),
                (end_x - perpx * half_width, end_y - perpy * half_width),
                (start_x - perpx * half_width, start_y - perpy * half_width)
            ]
            
            # ציור קטע הדרך
            pygame.draw.polygon(screen, (100, 100, 100), points)  # אפור
            
            # ציור שולי הדרך
            pygame.draw.line(screen, (255, 255, 255), 
                            (start_x + perpx * half_width, start_y + perpy * half_width),
                            (end_x + perpx * half_width, end_y + perpy * half_width), 2)
            pygame.draw.line(screen, (255, 255, 255),
                            (start_x - perpx * half_width, start_y - perpy * half_width),
                            (end_x - perpx * half_width, end_y - perpy * half_width), 2)
            
            # ציור קו אמצע מקווקו
            for i in range(10):
                t_start = i / 10
                t_end = (i + 0.6) / 10  # קו ארוך יותר מרווח
                
                if i % 2 == 0:  # רק קווים זוגיים
                    line_start_x = start_x + dx * (length * t_start)
                    line_start_y = start_y + dy * (length * t_start)
                    line_end_x = start_x + dx * (length * t_end)
                    line_end_y = start_y + dy * (length * t_end)
                    
                    pygame.draw.line(screen, (255, 255, 0),
                                    (line_start_x, line_start_y),
                                    (line_end_x, line_end_y), 2)
    
    def check_road_generation(self, car_x, car_y, min_segments_ahead=5):
        """
        בודק אם יש צורך ליצור קטעי דרך נוספים
        
        Args:
            car_x, car_y: מיקום המכונית
            min_segments_ahead: מספר קטעים מינימלי לפני המכונית
            
        Returns:
            True אם נוצרו קטעים חדשים
        """
        # אם אין קטעי דרך, יוצרים את הראשון
        if not self.road_segments:
            self.generate_road_segment(5)
            return True
        
        # מציאת הקטע הקרוב ביותר למכונית
        closest_segment_index = 0
        min_distance = float('inf')
        
        for i, segment in enumerate(self.road_segments):
            # חישוב מרחק מהמכונית למרכז הקטע
            segment_center_x = (segment.start[0] + segment.end[0]) / 2
            segment_center_y = (segment.start[1] + segment.end[1]) / 2
            
            distance = math.sqrt(
                (car_x - segment_center_x)**2 + 
                (car_y - segment_center_y)**2
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_segment_index = i
        
        # בדיקה כמה קטעים נותרו לפני המכונית
        segments_ahead = len(self.road_segments) - closest_segment_index - 1
        
        # אם יש פחות מהמינימום הנדרש, יוצרים חדשים
        if segments_ahead < min_segments_ahead:
            segments_to_add = min_segments_ahead - segments_ahead
            self.generate_road_segment(segments_to_add)
            return True
        
        return False
    
    def road_point_distance(self, x, y, include_width=True):
        """
        מחשב את המרחק מנקודה עד לדרך הקרובה ביותר
        
        Args:
            x, y: קואורדינטות הנקודה
            include_width: האם להתחשב ברוחב הדרך
            
        Returns:
            המרחק לדרך הקרובה והקטע הקרוב
        """
        if not self.road_segments:
            return float('inf'), None
        
        min_distance = float('inf')
        closest_segment = None
        
        for segment in self.road_segments:
            # חישוב מרחק מקטע קו
            start_x, start_y = segment.start
            end_x, end_y = segment.end
            
            # וקטור הקטע
            segment_dx = end_x - start_x
            segment_dy = end_y - start_y
            segment_length_sq = segment_dx**2 + segment_dy**2
            
            if segment_length_sq == 0:  # נקודת התחלה וסיום זהות
                distance = math.sqrt((x - start_x)**2 + (y - start_y)**2)
            else:
                # הטלה של הנקודה על הקו
                t = max(0, min(1, ((x - start_x) * segment_dx + 
                                   (y - start_y) * segment_dy) / 
                                  segment_length_sq))
                
                # נקודת הטלה על הקו
                proj_x = start_x + t * segment_dx
                proj_y = start_y + t * segment_dy
                
                # מרחק מהנקודה לנקודת ההטלה
                distance = math.sqrt((x - proj_x)**2 + (y - proj_y)**2)
            
            # אם מתחשבים ברוחב, מופחת רוחב הדרך
            if include_width and distance < segment.width / 2:
                distance = 0  # הנקודה על הדרך
            elif include_width:
                distance -= segment.width / 2
                distance = max(0, distance)  # לא מרחק שלילי
            
            if distance < min_distance:
                min_distance = distance
                closest_segment = segment
        
        return min_distance, closest_segment
