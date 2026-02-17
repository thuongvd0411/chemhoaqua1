import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import cv2
import mediapipe as mp
import av
import time
import os
import numpy as np
import threading
from collections import deque

# Import New Modules
from game_config import *
from game_engine import FruitGame, MoleGame
from renderer import GameRenderer
from data_manager import DataManager

# --- Shared State ---
class SharedGameState:
    def __init__(self):
        self.game = None
        self.lock = threading.Lock()
        
_shared_state = SharedGameState()

# --- Global State ---
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

class MotionTracker:
    def __init__(self, max_history=10, prediction_frames=2):
        self.history = deque(maxlen=max_history)
        self.max_history = max_history
        self.prediction_frames = prediction_frames
        self.last_valid_pos = None
        self.velocity = (0, 0)
        self.missing_frames = 0
        self.prediction_frames = 5 # Increased from default (likely 2-3) to 5 for better gap filling
        
    def update(self, pos):
        """
        Update with a new detected position (x, y).
        """
        if pos is not None:
            # Implement EMA Smoothing
            # alpha = 0.5 (Higher = more responsive, Lower = smoother)
            alpha = 0.5
            
            if self.last_valid_pos is not None:
                # Smooth Position
                sx = int(alpha * pos[0] + (1 - alpha) * self.last_valid_pos[0])
                sy = int(alpha * pos[1] + (1 - alpha) * self.last_valid_pos[1])
                smoothed_pos = (sx, sy)
                
                # Calculate velocity based on smoothed pos
                vx = smoothed_pos[0] - self.last_valid_pos[0]
                vy = smoothed_pos[1] - self.last_valid_pos[1]
                
                # Simple smoothing for velocity
                self.velocity = (vx * 0.7 + self.velocity[0] * 0.3, 
                                 vy * 0.7 + self.velocity[1] * 0.3)
                                 
                self.last_valid_pos = smoothed_pos
                self.history.append(smoothed_pos)
            else:
                # First point, no smoothing possible yet
                self.last_valid_pos = pos
                self.history.append(pos)
                
            self.missing_frames = 0
        else:
            # Prediction logic
            self.missing_frames += 1
            if self.missing_frames <= self.prediction_frames and self.last_valid_pos is not None:
                # Predict next point based on velocity
                pred_x = int(self.last_valid_pos[0] + self.velocity[0])
                pred_y = int(self.last_valid_pos[1] + self.velocity[1])
                
                # bounds check (0 to GAME_WIDTH/HEIGHT) would be good but not strictly necessary for logic
                self.last_valid_pos = (pred_x, pred_y)
                self.history.append((pred_x, pred_y))
            else:
                # Prediction expired -> Clear history to prevent stuck trails
                self.history.clear()
                self.last_valid_pos = None
                self.velocity = (0, 0)

    def get_segments(self):
        """
        Returns a list of line segments ((x1,y1), (x2,y2)) from the recent history.
        Used for collision detection.
        """
        if len(self.history) < 2:
            return []
        
        segments = []
        list_hist = list(self.history)
        for i in range(len(list_hist) - 1):
            segments.append((list_hist[i], list_hist[i+1]))
        return segments

    def get_trail(self):
        return list(self.history)
    
    def reset(self):
        self.history.clear()
        self.last_valid_pos = None
        self.velocity = (0, 0)

class GameVideoProcessor(VideoProcessorBase):
    def __init__(self):
        # AI Setup (Optimized)
        # Increase max_num_hands to 4 for 2 players
        # Tweak: Increased min_detection to 0.6 to reduce "ghost hands"
        self.hands = mp_hands.Hands(max_num_hands=4, min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6, model_complexity=1)
        
        # Modules
        # SHARED STATE LINK
        with _shared_state.lock:
            if _shared_state.game is None:
                # Default init
                pass 
            self.game = _shared_state.game 
            
        self.renderer = GameRenderer()
        
        # Trackers (Pool of 4)
        self.trackers = [MotionTracker() for _ in range(4)]
        
        # Asset Cache
        self.images = {}
        self.load_assets()
        
        self.mode = "FRUIT" # FRUIT or MOLE
        
        # User Context
        self.username = None
        self.display_name = None
        self.data_manager = DataManager() 
        self.is_ranked = False
        self.is_ranked = False
        self.saved_game_over = False
        
        # Video Recording
        self.frame_buffer = deque(maxlen=900) # 30fps * 30s buffer (Keep last 30s of gameplay)
        self.is_recording = True
        self.video_saved = False
        self.save_error = None
        self.exit_requested = False
        self.latest_video_path = None

    def save_video(self):
        if not self.frame_buffer: return
        
        try:
            # Save to recordings folder
            rec_dir = "recordings"
            if not os.path.exists(rec_dir):
                os.makedirs(rec_dir)
                
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{rec_dir}/gameplay_{timestamp}.mp4"
            
            # Get dimensions from first frame
            first_frame = self.frame_buffer[0]
            h, w, c = first_frame.shape
            
            # Writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 30.0, (w, h))
            
            for f in self.frame_buffer:
                out.write(f)
                
            out.release()
            self.video_saved = True
            self.save_error = None
            self.latest_video_path = filename
            print(f"Video saved to {filename}")
        except Exception as e:
            print(f"Warning: Could not save video: {e}")
            self.save_error = str(e)
            self.video_saved = False

    def load_assets(self):
        for k in ASSET_LABELS.keys():
            path = os.path.join(ASSETS_DIR, f"{k}.png")
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                # Resize ONCE on load to avoid runtime lag
                if img is not None:
                    h, w = img.shape[:2]
                    target_size = 80 # Standard size
                    if max(h, w) > target_size:
                        s = target_size / max(h, w)
                        img = cv2.resize(img, (int(w*s), int(h*s)))
                    self.images[k] = img

    def set_user_context(self, username, display_name=None):
        self.username = username
        self.display_name = display_name if display_name else username
        
    def set_game_mode(self, mode):
        # Reset trackers when switching modes
        # Reset trackers when switching modes
        if self.mode != mode: 
            for t in self.trackers: t.reset()
            self.frame_buffer.clear()
            self.video_saved = False
            
        self.mode = mode 
        
        if "FRUIT" in mode:
            if not isinstance(self.game, FruitGame):
                self.game = FruitGame(DEFAULT_TARGET_SCORE, DEFAULT_MAX_LIVES)
            self.is_ranked = ("RANKED" in mode)
        else:
            if not isinstance(self.game, MoleGame):
                self.game = MoleGame(DEFAULT_TARGET_SCORE, DEFAULT_MAX_LIVES)
            self.is_ranked = False
            
        # Update Shared State
        with _shared_state.lock:
             _shared_state.game = self.game

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            image = frame.to_ndarray(format="bgr24")
            
            # Ensure correct resolution
            image = cv2.resize(image, (GAME_WIDTH, GAME_HEIGHT))
            image = cv2.flip(image, 1)
            
            # --- Advanced Preprocessing (CLAHE) ---
            # 1. Sharpening (Keep this, it's cheap)
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            # image = cv2.filter2D(image, -1, kernel) # Optional: Can over-sharpen noise
            
            # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Convert to LAB to operate on Lightness channel
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            
            # Merge and convert back
            limg = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # Use 'enhanced' for AI detection, 'image' for display (or enhanced if we want user to see it)
            # Let's use enhanced for AI, and raw for display to avoid looking "processed"
            # BUT, if we draw trails on raw image, they match.
            
            if self.game is None:
                self.set_game_mode("FRUIT")

            # 1. AI Processing 
            detected_coords = [] # List of (x, y)
            
            # Always run AI even in Game Over for interaction (Restart button)
            # Use enhanced image for detection
            rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
            if "FRUIT" in self.mode:
                res = self.hands.process(rgb)
                if res.multi_hand_landmarks:
                    h, w, _ = image.shape
                    for lm in res.multi_hand_landmarks:
                        point = lm.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                        cx, cy = int(point.x * w), int(point.y * h)
                        detected_coords.append((cx, cy))
            else:
                res = self.pose.process(rgb)
                if res.pose_landmarks:
                    h, w, _ = image.shape
                    lm = res.pose_landmarks.landmark
                    lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
                    rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
                    detected_coords.append((int(lw.x * w), int(lw.y * h)))
                    detected_coords.append((int(rw.x * w), int(rw.y * h)))

            # Robust Tracking Matching (Greedy Euclidean to Predicted Pos)
            # Match detected points to existing trackers
            
            # 1. Get Predicted Positions for matching reference
            # We don't update trackers yet, just want to know where they *should* be.
            tracker_preds = []
            for t in self.trackers:
                if t.last_valid_pos is not None:
                    # Simple linear pred for matching
                    pred_x = t.last_valid_pos[0] + t.velocity[0]
                    pred_y = t.last_valid_pos[1] + t.velocity[1]
                    tracker_preds.append((pred_x, pred_y))
                else:
                    tracker_preds.append(None)

            # 2. Match
            assigned_trackers = set()
            matches = [] # (dist, p_idx, t_idx)
            
            for p_idx, (px, py) in enumerate(detected_coords):
                for t_idx, t_pred in enumerate(tracker_preds):
                    if t_idx in assigned_trackers: continue
                    
                    if t_pred is not None:
                        dist = np.hypot(px - t_pred[0], py - t_pred[1])
                        matches.append((dist, p_idx, t_idx))
                    else:
                        # Tracker is empty/reset. 
                        # Assign if it's free. Prioritize strict matches first.
                        matches.append((9999, p_idx, t_idx))

            matches.sort(key=lambda x: x[0])
            
            used_points = set()
            tracker_updates = {} # t_idx -> pos
            
            for dist, p_idx, t_idx in matches:
                if p_idx in used_points: continue
                if t_idx in assigned_trackers: continue
                
                # Stricter Threshold: 300px (Increased from 150 for fast movement)
                # If > 300px, it's likely a swap or new hand, don't snap old tracker.
                if dist < 300: 
                    tracker_updates[t_idx] = detected_coords[p_idx]
                    used_points.add(p_idx)
                    assigned_trackers.add(t_idx)

            # Rescue Pass: Assign remaining points to unmatched ACTIVE trackers (ignoring strict threshold)
            # This prevents ghosting where a fast hand breaks track and starts a new one concurrently.
            for p_idx in range(len(detected_coords)):
                if p_idx in used_points: continue
                
                best_t = -1
                best_dist = 9999
                
                for t_idx, t in enumerate(self.trackers):
                    if t_idx not in assigned_trackers and t.last_valid_pos is not None:
                        # Calculate distance to LAST KNOWN pos (not predicted, which might be wrong)
                        px, py = detected_coords[p_idx]
                        tx, ty = t.last_valid_pos
                        dist = np.hypot(px - tx, py - ty)
                        
                        if dist < best_dist:
                            best_dist = dist
                            best_t = t_idx
                
                # Assign if found (Rescue)
                if best_t != -1:
                    tracker_updates[best_t] = detected_coords[p_idx]
                    assigned_trackers.add(best_t)
                    used_points.add(p_idx)

            # Assign remaining points to empty trackers
            for p_idx in range(len(detected_coords)):
                if p_idx not in used_points:
                    # Find a free tracker
                    for t_idx, t in enumerate(self.trackers):
                        if t_idx not in assigned_trackers and t.last_valid_pos is None:
                            tracker_updates[t_idx] = detected_coords[p_idx]
                            assigned_trackers.add(t_idx)
                            used_points.add(p_idx)
                            break

            # Apply updates
            for t_idx, t in enumerate(self.trackers):
                if t_idx in tracker_updates:
                    t.update(tracker_updates[t_idx])
                else:
                    t.update(None) # Predict or Clear

            # Collect data for game
            interaction_segments = []
            active_trails = []
            for t in self.trackers:
                interaction_segments.extend(t.get_segments())
                if t.history:
                    active_trails.append(list(t.history))

            # 2. Game Logic Update
            # Spawn logic inside game depends on available types.
            # We should filter types here or update game logic to check existence
            # Rely on game_engine internal check if we assume it knows about ASSET_LABELS mapping
            
            # Pass segments even if game_over to allow "Button Logic"
            # However, update() handles the game over check.
            
            # Filter Spawning here? 
            # The Game Engine needs "available types" list for spawning.
            # If we passed it in __init__, great. If not, we should set it.
            # HACK: Set it directly on the instance if it's FRUIT
            if isinstance(self.game, FruitGame) and not hasattr(self.game, 'assets_configured'):
                 available_keys = [k for k in ASSET_LABELS.keys() if k in self.images]
                 self.game.fruit_types = available_keys
                 self.game.assets_configured = True
                 
            # PASS ACTIVE TRAILS
            # ALWAYS Update Game (for physics/particles even in Game Over)
            self.game.update(interaction_segments, active_trails=active_trails)
            
            if self.game.game_over:
                 action = self.game.check_game_over_interaction(interaction_segments)
                 if action == "RESTART":
                     self.game.reset()
                     self.saved_game_over = False
                     self.video_saved = False
                     self.save_error = None
                     self.latest_video_path = None
                     self.frame_buffer.clear()
                 elif action == "SAVE":
                     print("DEBUG: SAVE CLICKED")
                     if not self.video_saved:
                         self.save_video()
                 elif action == "EXIT":
                     print("DEBUG: EXIT CLICKED")
                     self.exit_requested = True

            # Audio Events Sync (REMOVED)
            with _shared_state.lock:
                 if hasattr(self.game, 'events') and self.game.events:
                     # _shared_state.audio_queue.extend(self.game.events) # Deleted queue
                     self.game.events.clear() # Consume events but do nothing

            # SAVE SCORE Logic
            if self.game.game_over and self.is_ranked and self.username and not self.saved_game_over:
                if self.game.won:
                    self.data_manager.add_stars(self.username, self.game.score)
                    self.saved_game_over = True
            
            if not self.game.game_over:
                self.saved_game_over = False

            # 3. Rendering
            # Draw Visual Hand Cursor (New Big Bright Circle)
            for t in self.trackers:
                if t.last_valid_pos:
                    image = self.renderer.draw_hand_cursor(image, int(t.last_valid_pos[0]), int(t.last_valid_pos[1]))

            # Draw Game Objects
            if "FRUIT" in self.mode:
                # Draw Halves First (behind whole fruits)
                for h in self.game.halves:
                    x, y = int(h['pos'][0]), int(h['pos'][1])
                    img = self.images.get(h['type'])
                    if img is not None:
                        # Draw both halves? No, the 'h' object is a specific half
                        image = self.renderer.draw_half_fruit(image, img, x, y, h['angle'], h['side'])
                        
                # Draw Whole Fruits
                for f in self.game.fruits:
                    x, y = int(f['pos'][0]), int(f['pos'][1])
                    img = self.images.get(f['type'])
                    if img is not None:
                        image = self.renderer.overlay_image(image, img, x, y, f['angle'])
                    else:
                         cv2.circle(image, (x, y), 30, (0, 0, 255), -1)

                # Screen Shake Effect
                if self.game.shake_timer > 0:
                    shake_x = np.random.randint(-15, 15)
                    shake_y = np.random.randint(-15, 15)
                    M = np.float32([[1, 0, shake_x], [0, 1, shake_y]])
                    image = cv2.warpAffine(image, M, (GAME_WIDTH, GAME_HEIGHT))

                # Red Flash Effect
                if self.game.flash_timer > 0 or (hasattr(self.game, 'frenzy_mode') and self.game.frenzy_mode):
                    overlay = image.copy()
                    
                    if hasattr(self.game, 'frenzy_mode') and self.game.frenzy_mode:
                         overlay[:] = (0, 255, 255) # Yellow (BGR)
                         alpha = 0.2 # Subtle Tint
                    else:
                         overlay[:] = (0, 0, 255) # Red
                         alpha = 0.3 * (self.game.flash_timer / 10.0) # Fade out
                    
                    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

                # Draw Auto Slashes
                if hasattr(self.game, 'auto_slashes'):
                    for s in self.game.auto_slashes:
                        cv2.line(image, s['p1'], s['p2'], (255, 255, 255), 3, cv2.LINE_AA)
                        # Add glow
                        cv2.line(image, s['p1'], s['p2'], (200, 255, 255), 8, cv2.LINE_AA)

            # Draw Particles
            image = self.renderer.draw_particles(image, self.game.particles)
            
            # Draw Trails
            if "FRUIT" in self.mode:
                for t in self.trackers:
                    trail = t.get_trail()
                    image = self.renderer.draw_trail(image, trail, (0, 255, 255))

            # Draw UI
            image = self.renderer.draw_text(image, f"ĐIỂM: {self.game.score}", 10, 10, 30, (0, 255, 0), outline_color=(0,0,0), outline_width=2)
            image = self.renderer.draw_text(image, f"MẠNG: {self.game.lives}", 10, 50, 30, (0, 0, 255), outline_color=(0,0,0), outline_width=2)
            
            if hasattr(self.game, 'combo_text_timer') and self.game.combo_text_timer > 0 and self.game.combo_count >= 3:
                 # Combo Text - Moved to Left Side, Smaller
                 txt = f"{self.game.combo_count} COMBO!"
                 color = (0, 165, 255) # Orange
                 # x=10, y=90 (below Lives)
                 image = self.renderer.draw_text(image, txt, 10, 90, 30, color, outline_color=(0,0,0), outline_width=2)
                 
                 # Optional: Show +Score multiplier small next to it
                 # image = self.renderer.draw_text(image, f"+{self.game.combo_count}", 200, 90, 30, (0, 255, 255), outline_color=(0,0,0), outline_width=2)

            if hasattr(self.game, 'frenzy_mode') and self.game.frenzy_mode:
                # Frenzy Text - Moved to Right Side or Top Center but smaller
                # Let's put it top center but smaller: size 25
                image = self.renderer.draw_text(image, "CHIÊU THỨC LIÊN HOÀN!", GAME_WIDTH//2 - 120, 30, 25, (0, 255, 255), outline_color=(0,0,255), outline_width=2)

            if self.display_name:
                 image = self.renderer.draw_text(image, f"{self.display_name}", GAME_WIDTH - 200, 10, 20, (255, 255, 255), outline_color=(0,0,0), outline_width=1)

            # Countdown / Game Over Overlay
            if self.game.is_counting_down:
                txt = self.game.get_countdown_text()
                text_size = 100 if txt != "BẮT ĐẦU!" else 70
                cx = GAME_WIDTH // 2 - (len(txt)*25)
                image = self.renderer.draw_text(image, txt, cx, GAME_HEIGHT//2, text_size, (0, 255, 255), outline_color=(0,0,0), outline_width=4)
                
            if self.game.game_over:
                res_txt = "CHIẾN THẮNG!" if self.game.won else "THUA CUỘC!"
                color = (0, 255, 0) if self.game.won else (0, 0, 255)
                # Center text
                image = self.renderer.draw_text(image, res_txt, GAME_WIDTH//2 - 180, GAME_HEIGHT//2 - 60, 70, color, outline_color=(255,255,255), outline_width=3)
                
                # Check delay for buttons
                if self.game.game_over_start_time and time.time() - self.game.game_over_start_time > 3.0:
                    btn_w, btn_h = 160, 60
                    gap = 20
                    start_x = (GAME_WIDTH - (btn_w * 3 + gap * 2)) // 2
                    y_center = GAME_HEIGHT // 2 + 80
                    
                    # Draw Buttons: [CHƠI LẠI] [LƯU VIDEO] [THOÁT]
                    image = self.renderer.draw_button(image, "CHƠI LẠI", start_x + btn_w//2, y_center, btn_w, btn_h)
                    
                    if hasattr(self, 'save_error') and self.save_error:
                        save_txt = "LỖI LƯU"
                        save_color = (0, 0, 255) # Red for error
                    else:
                        save_txt = "ĐÃ LƯU" if self.video_saved else "LƯU VIDEO"
                        save_color = None # Default
                        
                    image = self.renderer.draw_button(image, save_txt, start_x + btn_w + gap + btn_w//2, y_center, btn_w, btn_h)
                    
                    image = self.renderer.draw_button(image, "THOÁT", start_x + 2*(btn_w + gap) + btn_w//2, y_center, btn_w, btn_h)
            
            # Buffer Frame for Recording (Only raw gameplay + overlay? Or just final frame?)
            # Best to record final frame to see UI.
            if not self.game.game_over:
                self.frame_buffer.append(image)
            elif not self.video_saved: # Keep recording end screen until saved or stopped
                 if len(self.frame_buffer) < self.frame_buffer.maxlen:
                     self.frame_buffer.append(image)

            return av.VideoFrame.from_ndarray(image, format="bgr24")
        except Exception as e:
            print(f"Error in recv: {e}")
            import traceback
            traceback.print_exc()
            try:
                # Attempt to return a frame with error text
                img = frame.to_ndarray(format="bgr24")
                img = cv2.resize(img, (GAME_WIDTH, GAME_HEIGHT))
                img = cv2.flip(img, 1) # FLIP TO MATCH GAMEPLAY
                
                cv2.putText(img, f"ERROR: {str(e)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            except:
                return frame

# --- Streamlit Main App ---
st.set_page_config(page_title="Be & An Game V4", layout="wide")

# Init Data Manager
if 'data_manager' not in st.session_state:
    st.session_state['data_manager'] = DataManager()
if 'user' not in st.session_state:
    st.session_state['user'] = None

# --- Helpers ---
def safe_rerun():
    """Rerun the app in a cross-version compatible way"""
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        # Fallback for very old versions? unlikely needed
        pass

def save_score_callback(score):
    """Callback to save score from processor"""
    user = st.session_state.get('user')
    if user:
        # Add stars/score
        dm = st.session_state['data_manager']
        dm.add_stars(st.session_state['username'], score)
        # Refresh user data
        st.session_state['user'] = dm.login(st.session_state['username'])
        st.toast(f"Đã lưu kết quả: +{score} sao!")

# --- Sidebar: Login / Profile ---
st.sidebar.title("Hồ sơ đấu thủ")

if st.session_state['user']:
    u = st.session_state['user']
    st.sidebar.success(f"Xin chào, {u['display_name']}!")
    
    # Show Avatar/Rank
    rank_name, rank_emoji, rank_color = u.get('rank_info', ("Gà mờ", "🥚", "#808080")) # Default fallback
    st.sidebar.markdown(f"**Hạng:** {rank_emoji} {rank_name}")
    st.sidebar.markdown(f"**Sao:** {u.get('stars', 0)} ⭐")
    
    # Upload Photo
    st.sidebar.markdown("---")
    st.sidebar.subheader("Cập nhật ảnh đại diện")
    uploaded_file = st.sidebar.file_uploader("Chọn ảnh (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
         # Save logic
         try:
             bytes_data = uploaded_file.read()
             # Save to assets with username
             with open(os.path.join(ASSETS_DIR, f"{st.session_state['username']}_avatar.png"), "wb") as f:
                 f.write(bytes_data)
             st.sidebar.success("Đã cập nhật ảnh!")
         except Exception as e:
             st.sidebar.warning(f"Không thể lưu ảnh (Read-only mode): {e}")
         
    if st.sidebar.button("Đăng xuất"):
        st.session_state['user'] = None
        safe_rerun()
else:
    tab1, tab2 = st.sidebar.tabs(["Đăng Nhập", "Đăng Ký"])
    
    with tab1:
        login_user = st.text_input("Tên đăng nhập")
        if st.button("Vào game"):
            user = st.session_state['data_manager'].login(login_user)
            if user:
                st.session_state['user'] = user
                st.session_state['username'] = login_user
                safe_rerun()
            else:
                st.error("Không tìm thấy người dùng!")
                
    with tab2:
        reg_user = st.text_input("Tên đăng nhập mới")
        reg_name = st.text_input("Tên hiển thị")
        if st.button("Tạo tài khoản"):
            success, msg = st.session_state['data_manager'].register(reg_user, reg_name)
            if success:
                st.success("Tạo thành công! Hãy đăng nhập.")
            else:
                st.error(msg)

# --- Main Content ---
st.title("Be & An Game - High Performance")

# Tabs
menu = ["Chơi Game", "Bảng Xếp Hạng", "Lịch Sử"]
choice = st.selectbox("Menu chính", menu, label_visibility="collapsed")

if choice == "Chơi Game":
    c1, c2 = st.columns([1, 3])
    
    with c1:
        st.subheader("Cài đặt")
        mode = st.radio("Chế độ", ["Chém Hoa Quả (Tập Luyện)", "Thử Thách (Tính Điểm)", "Đập Chuột"])
        
        # Mode Logic
        if mode == "Chém Hoa Quả (Tập Luyện)":
            game_key = "FRUIT"
            is_ranked = False
        elif mode == "Thử Thách (Tính Điểm)":
            game_key = "FRUIT_RANKED"
            is_ranked = True
        else:
            game_key = "MOLE"
            is_ranked = False
            
        if is_ranked and not st.session_state['user']:
            st.warning("Bạn cần đăng nhập để chơi chế độ Thử Thách!")
            
    with c2:
        # WebRTC Streamer
        # If ranked, we might want to pass user info to processor, 
        # but processor runs in separate thread. We can't pass 'user' dict directly easily if it changes.
        # But we can pass static config via factory args if we used a factory wrapper, 
        # or just set it after creation if using session state (unreliable across threads).
        
        # Simplest: The Processor class checks specific thread-safe queue or shared dict? 
        # streamlit-webrtc runs in a separate thread.
        # We will use the 'client_settings' or just basic mode setting.
        # For saving score to DB, the processor needs to callback or write to a shared file/db.
        # Since DataManager uses a file, we can write from the Processor thread safely enough for this scale.
        
        ctx = webrtc_streamer(
            key="game_stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            video_processor_factory=GameVideoProcessor,
            async_processing=True,
        )

        # Polling for Exit/Video events from Processor
        if ctx.state.playing:
            status_area = st.empty()
            while ctx.state.playing:
                if ctx.video_processor:
                    # Check Exit
                    if ctx.video_processor.exit_requested:
                        # status_area.warning("Đang thoát game...") # causes rerun itself?
                        st.session_state['force_exit'] = True
                        break
                    
                    # Check Video
                    if ctx.video_processor.latest_video_path:
                        st.session_state['last_video'] = ctx.video_processor.latest_video_path
                        ctx.video_processor.latest_video_path = None
                        st.success("Video đã lưu!") # Visual feedback
                        time.sleep(1)
                        st.rerun() # Force rerun to show video player
                
                time.sleep(0.5)

        # Handle Exit Flag (Outside Loop)
        if st.session_state.get('force_exit'):
            del st.session_state['force_exit']
            st.rerun() # This should reset the app state if not controlled by other session vars
            
        # Handle Video Playback
        if st.session_state.get('last_video'):
            st.divider()
            st.subheader("Xem lại Video vừa chơi")
            video_path = st.session_state['last_video']
            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.error("Không tìm thấy file video.")
            
            if st.button("Đóng Video"):
                del st.session_state['last_video']
                st.rerun()

        if ctx.video_processor:
            # Pass User Context to Processor
            if st.session_state['user']:
                # Pass both username (for ID) and display_name (for UI)
                ctx.video_processor.set_user_context(
                    st.session_state['username'], 
                    st.session_state['user'].get('display_name')
                )
            else:
                ctx.video_processor.set_user_context(None, None)
                
            ctx.video_processor.set_game_mode(game_key)

elif choice == "Bảng Xếp Hạng":
    st.subheader("🏆 Bảng Xếp Hạng")
    leaderboard = st.session_state['data_manager'].get_leaderboard()
    
    for i, p in enumerate(leaderboard):
        msg = f"#{i+1} **{p['display_name']}** - {p['stars']} ⭐"
        if i == 0:
            st.warning(f"🥇 {msg}")
        elif i == 1:
            st.success(f"🥈 {msg}")
        elif i == 2:
            st.info(f"🥉 {msg}")
        else:
            st.markdown(msg)

elif choice == "Lịch Sử":
    # Simple history view
    st.subheader("📜 Lịch Sử Đấu")
    if st.session_state['user']:
        # Reload to get latest
        user = st.session_state['data_manager'].login(st.session_state['username'])
        if user and 'history' in user and user['history']:
            # Create a nice table
            import pandas as pd
            df = pd.DataFrame(user['history'])
            # Sort by date desc
            df = df.iloc[::-1]
            
            # Format columns
            st.dataframe(
                df,
                column_config={
                    "date": "Thời gian",
                    "score": "Điểm số ⭐️"
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("Chưa có lịch sử đấu.")
    else:
        st.warning("Vui lòng đăng nhập để xem lịch sử.")
