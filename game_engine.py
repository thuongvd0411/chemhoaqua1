import random
import time
import numpy as np
from game_config import GAME_WIDTH, GAME_HEIGHT

class Particle:
    def __init__(self, x, y, color, is_splash=False):
        self.x = x
        self.y = y
        self.life = 0.8
        self.color = color
        self.size = random.randint(2, 5)
        if is_splash:
            self.vx = random.uniform(-3, 3)
            self.vy = random.uniform(-8, -4)
        else:
            self.vx = random.uniform(-5, 5)
            self.vy = random.uniform(-5, 5)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.5 # Gravity
        self.life -= 0.05

class BaseGame:
    def __init__(self, target_score=20, max_lives=5):
        self.score = 0
        self.max_lives = max_lives
        self.lives = max_lives
        self.target_score = target_score
        self.game_over = False
        self.won = False
        self.particles = []
        self.events = [] # For Audio/External triggers
        
        # Countdown
        self.is_counting_down = True
        self.countdown_start = time.time()
        self.countdown_duration = 3.0
        self.game_over_start_time = None
        
        # Config
        self.width = GAME_WIDTH
        self.height = GAME_HEIGHT

    def update_countdown(self):
        if not self.is_counting_down: return
        if time.time() - self.countdown_start > self.countdown_duration:
            self.is_counting_down = False

    def get_countdown_text(self):
        remaining = self.countdown_duration - (time.time() - self.countdown_start)
        if remaining > 0:
            return str(int(np.ceil(remaining)))
        return "BẮT ĐẦU!"

    def check_game_over_interaction(self, interaction_segments):
        # Enforce 3s delay
        if self.game_over_start_time is None: return None
        if time.time() - self.game_over_start_time < 3.0: return None

        # Button Config
        # Center X = self.width // 2
        # Buttons: [CHƠI LẠI] [LƯU KQ] [THOÁT]
        # Width: 160 each, Gap: 20 -> Total 520
        # Start X = (self.width - 520) // 2 -> 60
        
        btn_w, btn_h = 160, 60
        gap = 20
        start_x = (self.width - (btn_w * 3 + gap * 2)) // 2
        y_center = self.height // 2 + 80
        
        buttons = [
            {"label": "RESTART", "x": start_x + btn_w//2, "y": y_center},
            {"label": "SAVE", "x": start_x + btn_w + gap + btn_w//2, "y": y_center},
            {"label": "EXIT", "x": start_x + 2*(btn_w + gap) + btn_w//2, "y": y_center}
        ]
        
        for p1, p2 in interaction_segments:
            for btn in buttons:
                bx, by = btn['x'], btn['y']
                x1, x2 = bx - btn_w//2, bx + btn_w//2
                y1, y2 = by - btn_h//2, by + btn_h//2
                
                # Check interaction (simple box overlap with segment end p2)
                if (x1 <= p2[0] <= x2 and y1 <= p2[1] <= y2):
                    return btn['label']
        return None

    def reset(self):
        self.score = 0
        self.lives = self.max_lives
        self.game_over = False
        self.won = False
        self.particles = []
        self.events = []
        self.is_counting_down = True
        self.countdown_start = time.time()
        self.game_over_start_time = None


class FruitGame(BaseGame):
    def __init__(self, target_score=20, max_lives=5):
        super().__init__(target_score, max_lives)
        self.fruits = []
        self.halves = [] # New: Track split fruit parts
        self.last_spawn = time.time()
        self.spawn_interval = 1.5
        self.fruit_types = ['apple', 'orange', 'mango', 'watermelon', 'bomb', 'spike', 'first_aid']
        
        # Bomb Effects
        self.shake_timer = 0
        self.flash_timer = 0
        
        # Frenzy Mode (Hidden Skill)
        self.frenzy_mode = False
        self.frenzy_timer = 0
        self.frenzy_duration = 3.0 # Seconds
        
        # New Skill: Merge Hands
        self.merge_timer = 0
        self.auto_slashes = [] # List of (p1, p2, timer)

    def reset(self):
        super().reset()
        self.fruits = []
        self.halves = []
        self.last_spawn = time.time()
        self.frenzy_mode = False
        self.frenzy_timer = 0
        self.merge_timer = 0
        self.auto_slashes = []
        self.shake_timer = 0
        self.flash_timer = 0

    def spawn(self, available_types):
        if not available_types: return
        
        # Filter Logic
        types_to_spawn = [t for t in available_types]
        
        # 1. Reduce Bomb Probability (20% less than normal)
        # or just make it explicit:
        # If frenzy mode, NO BOMBS, only fruit!
        if self.frenzy_mode:
            types_to_spawn = [t for t in types_to_spawn if t not in ['bomb', 'spike']]
            if not types_to_spawn: return # Should not happen if fruits exist
            
            # Spawn logic for Frenzy: Multiple fruits!
            # Reduced from (2, 4) to (1, 3) for ~20-25% reduction on average
            count = random.randint(1, 3)
            for _ in range(count):
                self._spawn_single(types_to_spawn)
        else:
            # Normal Spawn
            # Weighted choice: Fruits 80%, Bombs 20% -> Reduce bomb by 20% relative?
            # Let's say original was uniform. 
            # We want strictly less bombs.
            pass # We'll just use a weighted random choice helper or simple logic
            
            # Simple Logic: 15% chance for bomb, rest fruit (if bomb exists)
            should_spawn_bomb = False
            if 'bomb' in types_to_spawn and random.random() < 0.15: # Reduced from implicit ~1/6 (16%) -> 15% isn't much reduction.
                 # Actually 1/7 is ~14%. User said "reduce by 20%".
                 # Let's assume default was ~20% total bad items. New is ~16%.
                 pass

            # Just pick random from list but filter 'bomb' out first, then decide to add it back
            safe_types = [t for t in types_to_spawn if t not in ['bomb', 'spike']]
            bad_types = [t for t in types_to_spawn if t in ['bomb', 'spike']]
            
            final_type = None
            if bad_types and random.random() < 0.15: # 15% chance for bad
                final_type = random.choice(bad_types)
            elif safe_types:
                final_type = random.choice(safe_types)
                
            if final_type:
                 self.fruits.append({
                    'type': final_type,
                    'pos': [random.randint(100, self.width-100), self.height + 20],
                    'vel': [random.uniform(-4, 4), random.uniform(-22, -28)],
                    'hit': False,
                    'angle': 0, 
                    'ang_vel': random.uniform(-5, 5)
                })

    def check_merge_gesture(self, active_trails):
        # Need exactly 2 hands (or more, but we check 2)
        if len(active_trails) < 2: 
            self.merge_timer = 0
            return

        # Get latest points
        p1 = active_trails[0][-1]
        p2 = active_trails[1][-1]
        
        # Calculate distance
        dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
        
        # Threshold: 60px (Hands touching/merged)
        if dist < 60:
            self.merge_timer += (1/30.0) # Assume ~30fps
            if self.merge_timer >= 1.0: # Hold for 1s
                self.activate_frenzy("HỢP THỂ KÍCH HOẠT!")
                self.merge_timer = 0
        else:
            self.merge_timer = 0

    def check_special_moves(self, active_trails):
        # REPLACED by Merge Gesture as per request
        self.check_merge_gesture(active_trails)


    def activate_frenzy(self, text):
        if self.frenzy_mode: return
        self.frenzy_mode = True
        self.frenzy_mode = True
        self.frenzy_timer = 3.0 # 3 seconds
        self.flash_timer = 5
        self.flash_timer = 5
        self.events.append("frenzy") # Trigger Sound
        
        # Visual Flux
        for _ in range(20):
             self.particles.append(Particle(random.randint(0, self.width), random.randint(0, self.height), (255, 255, 255)))

    def _spawn_single(self, type_list):
        t = random.choice(type_list)
        self.fruits.append({
            'type': t,
            'pos': [random.randint(100, self.width-100), self.height + 20],
            'vel': [random.uniform(-4, 4), random.uniform(-22, -28)],
            'hit': False,
            'angle': 0, 
            'ang_vel': random.uniform(-5, 5)
        })

    def dist_point_to_segment(self, p, s1, s2):
        """
        Calculates the distance from point p to line segment s1-s2.
        p, s1, s2 are (x,y) tuples/arrays.
        """
        x, y = p
        x1, y1 = s1
        x2, y2 = s2
        
        px = x2 - x1
        py = y2 - y1
        
        if px == 0 and py == 0:
            return np.hypot(x - x1, y - y1)

        u = ((x - x1) * px + (y - y1) * py) / float(px * px + py * py)

        if u > 1:
            u = 1
        elif u < 0:
            u = 0

        dx = x1 + u * px
        dy = y1 + u * py

        return np.hypot(x - dx, y - dy)

    def update(self, interaction_segments, active_trails=None):
        """
        interaction_segments: list of ((x1, y1), (x2, y2)) tuples.
        active_trails: list of lists of (x,y) tuples representing the full path of each hand.
        """
        self.update_countdown()
        if self.is_counting_down: return

        # 1. Game Logic (Only if playing)
        if not self.game_over:
            # Update Effects
            if self.shake_timer > 0: self.shake_timer -= 1
            if self.flash_timer > 0: self.flash_timer -= 1
            
            # Update Frenzy Mode
            if self.frenzy_mode:
                self.frenzy_timer -= (1/30.0) # Approx 30fps
                self.shake_timer = 2 # Constant shake
                
                # Auto Slashes
                if random.random() < 0.5: # 50% chance per frame to generate slash
                    # Generate random slash line
                    if random.random() < 0.5:
                        # Horizontal-ish
                        y = random.randint(50, self.height-50)
                        p1 = (0, y + random.randint(-50, 50))
                        p2 = (self.width, y + random.randint(-50, 50))
                    else:
                        # Diagonal
                        p1 = (random.randint(0, self.width), 0)
                        p2 = (random.randint(0, self.width), self.height)
                    
                    self.auto_slashes.append({'p1': p1, 'p2': p2, 'life': 5}) # Lasts 5 frames
                
                # Update Auto Slashes life
                for s in self.auto_slashes:
                    s['life'] -= 1
                self.auto_slashes = [s for s in self.auto_slashes if s['life'] > 0]
                
                # Check Auto Slash Collisions
                for s in self.auto_slashes:
                    for f in self.fruits:
                        if not f['hit']:
                             dist = self.dist_point_to_segment(f['pos'], s['p1'], s['p2'])
                             if dist < 50:
                                 f['hit'] = True
                                 self.handle_hit(f)

                if self.frenzy_timer <= 0:
                    self.frenzy_mode = False
                    self.auto_slashes = []
                    self.spawn_interval = 1.0 # Reset to normal-ish
                    
            # Access Special Moves
            if active_trails:
                self.check_special_moves(active_trails)

            # Spawn
            current_interval = 0.2 if self.frenzy_mode else self.spawn_interval
            if time.time() - self.last_spawn > current_interval:
                self.spawn([t for t in self.fruit_types]) 
                self.last_spawn = time.time()
                if not self.frenzy_mode and self.spawn_interval > 0.6: 
                    self.spawn_interval *= 0.98

        # 2. Physics & Cleanup (Always Run)


        # Physics - Active Fruits
        for f in self.fruits:
            f['pos'][0] += f['vel'][0]
            f['pos'][1] += f['vel'][1]
            f['vel'][1] += 0.5 # Gravity
            f['angle'] += f['ang_vel']
            
        # Physics - Halves
        for h in self.halves:
            h['pos'][0] += h['vel'][0]
            h['pos'][1] += h['vel'][1]
            h['vel'][1] += 0.6 # Slightly heavier gravity for halves
            h['angle'] += h['ang_vel']

        # Cleanup
        self.fruits = [f for f in self.fruits if f['pos'][1] < self.height + 100 and not f['hit']]
        self.halves = [h for h in self.halves if h['pos'][1] < self.height + 100]

        # Collisions (Segment based) - Only if Playing
        if not self.game_over:
            for p1, p2 in interaction_segments:
                for f in self.fruits:
                    if not f['hit']:
                        # Check distance from fruit center to segment p1-p2
                        dist = self.dist_point_to_segment(f['pos'], p1, p2)
                        
                        if dist < 65: # Hit radius increased for better feel
                            f['hit'] = True
                            self.handle_hit(f)

        # Particles
        for p in self.particles: p.update()
        self.particles = [p for p in self.particles if p.life > 0]
        
        if self.lives <= 0: 
            if not self.game_over:
                self.game_over = True
                self.game_over_start_time = time.time()
        
        if self.score >= self.target_score: 
            self.won = True
            if not self.game_over:
                self.game_over = True
                self.game_over_start_time = time.time()

    def handle_hit(self, fruit):
        if fruit['type'] in ['bomb', 'spike']:
            if self.frenzy_mode:
                # In Frenzy, bombs behave like fruits (bonus points or just destroy safe)
                self.score += 5 # Bonus for destroying bombs in frenzy!
                self.events.append("hit_bonus")
                # Simple explosion effect but no damage
                for _ in range(20):
                     self.particles.append(Particle(fruit['pos'][0], fruit['pos'][1], (255, 255, 0)))
            else:
                self.lives -= 1
                self.events.append("hit_bomb")
                # Explosion particles
                for _ in range(40): # More particles
                    self.particles.append(Particle(fruit['pos'][0], fruit['pos'][1], (0, 0, 255))) # Red/Orange
                    self.particles.append(Particle(fruit['pos'][0], fruit['pos'][1], (0, 165, 255)))
                
                # Trigger Effects
                self.shake_timer = 20 # Shake for ~20 frames
                self.flash_timer = 10 # Flash for ~10 frames
            
        elif fruit['type'] == 'first_aid':
            self.lives = min(self.lives + 1, 10)
            self.events.append("hit_item")
        else:
            self.score += 1
            self.events.append("hit_fruit")
            # Juice particles
            # Juice particles
            color = (0, 255, 255) # Yellow default
            
            # IMPROVED HIT EFFECT: More particles, faster spread
            for _ in range(8):
                p = Particle(fruit['pos'][0], fruit['pos'][1], color, is_splash=True)
                p.vx *= 1.5
                p.vy *= 1.5
                self.particles.append(p)
                
            # Add a "Flash" particle (White, short life) to simulate impact
            flash = Particle(fruit['pos'][0], fruit['pos'][1], (255, 255, 255), is_splash=False)
            flash.life = 0.3
            flash.size = 15 # Big flash
            self.particles.append(flash)
            
            # Create Halves
            v = fruit['vel']
            base_x, base_y = fruit['pos']
            
            # Left Half
            self.halves.append({
                'type': fruit['type'],
                'pos': [base_x - 10, base_y],
                'vel': [v[0] - 5, v[1] - 2], 
                'angle': fruit['angle'],
                'ang_vel': random.uniform(-10, -5),
                'side': 'left'
            })
            
            # Right Half
            self.halves.append({
                'type': fruit['type'],
                'pos': [base_x + 10, base_y],
                'vel': [v[0] + 5, v[1] - 2],
                'angle': fruit['angle'],
                'ang_vel': random.uniform(5, 10),
                'side': 'right'
            })


class MoleGame(BaseGame):
    def __init__(self, target_score=20, max_lives=5):
        super().__init__(target_score, max_lives)
        self.holes = [(100, 400), (210, 400), (320, 400), (430, 400), (540, 400)]
        self.moles = []
        self.last_spawn = time.time()
        
    def reset(self):
        super().reset()
        self.moles = []
        self.last_spawn = time.time()
        
    def update(self, interaction_segments):
        self.update_countdown()
        if self.is_counting_down or self.game_over: return
        
        # Simple Mole spawn logic
        if time.time() - self.last_spawn > 1.5:
            idx = random.randint(0, 4)
            # Only spawn if hole is empty
            if not any(m['idx'] == idx for m in self.moles):
                self.moles.append({'idx': idx, 'spawn_time': time.time(), 'hit': False, 'type': 'mole'})
            self.last_spawn = time.time()
            
        self.moles = [m for m in self.moles if time.time() - m['spawn_time'] < 1.5]
        
        # Check collisions (Hammer hits)
        for p1, p2 in interaction_segments:
            # We use p2 (current pos) for mole hits usually, but segment works too
            # For mole, we check if the end point (hammer head) is near the mole
            # But "slashing" moles is also fun. Let's support both.
            for m in self.moles:
                if not m['hit']:
                    hx, hy = self.holes[m['idx']]
                    # Check if segment crosses the hole area
                    # Use simple point check for the "head" of the motion (p2) to simulate a "bonk"
                    # But if we want "slash" style, we use segment.
                    # Let's use simple distance to p2 for now to keeping "Whack" feel, 
                    # but maybe the user wants to slash them.
                    # Given the request is "slash logic", I'll use segment logic for robust detection.
                    
                    # Distance from hole center to slash segment
                    dist = np.hypot(p2[0] - hx, p2[1] - hy) 
                    # For mole, usually you "tap" it.
                    # Let's stick to point-radius at the end or start.
                    # Actually, if I move fast over it, it should hit.
                    dist_seg = self.dist_point_to_segment_mole((hx, hy), p1, p2)
                    
                    if dist_seg < 50:
                        m['hit'] = True
                        self.score += 1

        if self.lives <= 0: 
            if not self.game_over:
                self.game_over = True
                self.game_over_start_time = time.time()

        if self.score >= self.target_score: 
            self.won = True
            if not self.game_over:
                self.game_over = True
                self.game_over_start_time = time.time()
            
    def dist_point_to_segment_mole(self, p, s1, s2):
        # Re-implement or simple helper
        x, y = p
        x1, y1 = s1
        x2, y2 = s2
        px = x2 - x1
        py = y2 - y1
        if px == 0 and py == 0: return np.hypot(x - x1, y - y1)
        u = ((x - x1) * px + (y - y1) * py) / float(px * px + py * py)
        if u > 1: u = 1
        elif u < 0: u = 0
        dx = x1 + u * px
        dy = y1 + u * py
        return np.hypot(x - dx, y - dy)
