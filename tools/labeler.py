import cv2
import os
import json
import re
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from datetime import datetime
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch

# Optional pose-assist backend.
#
# ultralytics is AGPL, and a model trained or pre-filled with it inherits that.
# Models made in this app are meant to be shared and used by anyone, in any
# build, so the app itself never depends on it. This file is dev-only tooling
# under tools/ — not imported by the app, never pulled into the exe from
# main.py — and pose pre-fill only speeds up placing keypoints by hand.
#
# The import is optional so nothing in this repo hard-depends on an AGPL
# package: without ultralytics the labeller still runs, it just loses the pose
# pre-fill and every keypoint is placed by hand.
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def _disp(internal: str) -> str:
    """Strip the dedup suffix (~2, ~3 …) to get the display/class name."""
    if '~' in internal:
        base, _, suf = internal.rpartition('~')
        if suf.isdigit():
            return base
    return internal


def _make_internal(display: str, existing: list) -> str:
    """Return a unique internal key for *display*, suffixing ~2, ~3 … if needed."""
    if display not in existing:
        return display
    n = 2
    while f"{display}~{n}" in existing:
        n += 1
    return f"{display}~{n}"


def _pts(val):
    """Normalise a points-dict value to a list of (x, y) tuples.
    Handles old single-instance format [x, y] and new [[x1,y1],[x2,y2]]."""
    if not val:
        return []
    if isinstance(val[0], (int, float)):
        return [tuple(val)]          # old format: [x, y]
    return [tuple(p) for p in val]  # new format: [[x,y], ...]


def natural_sort_key(path):
    """Natural sort key for human-friendly sorting (e.g., 8 before 10)"""
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    
    filename = path.stem
    return [convert(c) for c in re.split('([0-9]+)', filename)]

class VideoLabelerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 视频标注器 - AI 辅助跟踪")
        self.root.geometry("1200x800")
        
        # State
        self.video_path = None
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.current_frame = 0
        self.points = {}
        self.occluded = set()  # keypoints explicitly marked hidden/inside this frame
        self.current_kp = 0
        self.labeled_frames = []
        self.is_playing = False
        self.slider_update = True
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.interpolation_mode = "tracker"
        
        # YOLO Tracker state
        self.yolo_model = None
        self.tracking_data = {}  # {frame: {keypoint: (x,y)}} - Stores successful coordinates
        self.tracking_active = False
        # Store history for prediction: {keypoint_name: [(frame, x, y), ...]}
        self.track_history = defaultdict(list) 
        self.tracking_method = "botsort"
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.5
        # Prediction & Smoothing state
        self.prediction_window_size = 3 # For extrapolating lost frames
        self.smoothing_window_size = 5  # Moving Average filter size (N frames)

        # Prediction variables
        self.prediction_window_size = 3 # How many frames back to use for prediction
        
        # Adjustment state
        self.adjusting_interpolated = False
        self.selected_interpolated_frame = None
        self.adjusting_keypoint = None
        
        # Spacebar state
        self.space_held = False
        
        # File navigation state
        self.video_files = []
        self.current_video_idx = 0
        self.progress_data = {}
        self.progress_file = Path("labeling_progress.json")
        
        # Track if current video has been exported
        self.current_video_exported = False
        
        # Keypoint definitions (user-defined, persisted to a config file)
        self.keypoint_config_file = Path("labeler_keypoints.json")
        self._keypoint_config_existed = self.keypoint_config_file.exists()
        self.keypoint_names = self._load_keypoint_names()
        self.keypoint_display_names = {}
        self.colors = {}      # BGR tuples for cv2 drawing
        self.color_hex = {}   # '#RRGGBB' for tk widgets
        self._rebuild_keypoint_meta()

        self.status_var = tk.StringVar(value="就绪。请选择关键点，然后在视频画面中点击放置。")
        
        self.load_progress()
        self.setup_ui()
        self.bind_shortcuts()
        self._finalize_window()
        self.load_yolo_model()

        # First run (no config yet): nudge the user toward the inline editor
        if not self._keypoint_config_existed and not self.keypoint_names:
            self.status_var.set("暂无关键点，请在 🎯 关键点面板中点击“➕ 添加关键点”。")

    # *************************************************************
    # *** KEYPOINT LABEL CONFIG (user-defined) ***
    # *************************************************************
    DEFAULT_KEYPOINTS = []  # none shipped in code; defined by the user on first run

    def _load_keypoint_names(self):
        """Load keypoint names from config, converting duplicate display names to
        unique internal keys (source_a, source_a~2, source_a~3 …)."""
        try:
            if self.keypoint_config_file.exists():
                with open(self.keypoint_config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                raw = [str(n).strip() for n in data.get('keypoint_names', []) if str(n).strip()]
                if raw:
                    internal = []
                    for display in raw:
                        internal.append(_make_internal(display, internal))
                    return internal
        except Exception as e:
            print(f"⚠️ 无法读取 {self.keypoint_config_file}：{e}")
        return list(self.DEFAULT_KEYPOINTS)

    def _save_keypoint_names(self):
        """Persist display names (strips ~N dedup suffix) to the config file."""
        try:
            with open(self.keypoint_config_file, 'w', encoding='utf-8') as f:
                json.dump({'keypoint_names': [_disp(n) for n in self.keypoint_names]}, f, indent=2)
        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存关键点配置：\n{e}")

    def _rebuild_keypoint_meta(self):
        """Regenerate display names + colors. Duplicate display names share a color."""
        import colorsys
        self.keypoint_display_names = {}
        self.colors = {}
        self.color_hex = {}
        unique_display = list(dict.fromkeys(_disp(n) for n in self.keypoint_names))
        n = max(len(unique_display), 1)
        disp_color = {}
        for i, disp in enumerate(unique_display):
            r, g, b = colorsys.hsv_to_rgb(i / n, 0.85, 1.0)
            R, G, B = int(r * 255), int(g * 255), int(b * 255)
            disp_color[disp] = ((B, G, R), f'#{R:02X}{G:02X}{B:02X}')
        for name in self.keypoint_names:
            disp = _disp(name)
            self.colors[name] = disp_color[disp][0]
            self.color_hex[name] = disp_color[disp][1]
            self.keypoint_display_names[name] = disp.replace('_', ' ').title()

    def _add_keypoint(self):
        """Append a new keypoint with a unique default name; ready to rename."""
        existing = set(self.keypoint_names)
        i = 1
        while f"label{i}" in existing:
            i += 1
        new_name = f"label{i}"
        self.keypoint_names.append(new_name)
        self._rebuild_keypoint_meta()
        self._save_keypoint_names()
        self._populate_keypoint_buttons()
        self.select_keypoint(new_name)
        # Focus its name field so the user can type a real name immediately
        entry = self.kp_buttons.get(new_name, {}).get('entry')
        if entry:
            entry.focus_set()
            entry.select_range(0, tk.END)
        self.status_var.set(f"➕ 已添加“{new_name}”，请输入名称并按 Enter")

    def _remove_keypoint(self, name):
        """Remove a keypoint and strip its points so data stays consistent."""
        if name not in self.keypoint_names:
            return
        self.keypoint_names.remove(name)
        for f in self.labeled_frames:
            f.get('points', {}).pop(name, None)
        self.points.pop(name, None)
        self._rebuild_keypoint_meta()
        self._save_keypoint_names()
        self._populate_keypoint_buttons()
        self.current_kp = 0
        if self.keypoint_names:
            self.select_keypoint(self.keypoint_names[0])
        if self.cap:
            self.show_frame(self.current_frame)
        self.status_var.set(f"🗑️ 已删除关键点“{name}”")

    def _commit_rename(self, old, new):
        """Rename a keypoint in place, migrating any existing labeled points."""
        new = (new or "").strip()
        if old not in self.keypoint_names or new == old:
            return
        if not new:
            self.kp_buttons[old]['name_var'].set(old)  # revert blank
            return
        # Allow duplicate display names — generate a unique internal key
        new_internal = _make_internal(new, [k for k in self.keypoint_names if k != old])
        self.keypoint_names[self.keypoint_names.index(old)] = new_internal
        new = new_internal
        # Migrate points so renaming doesn't orphan existing labels
        for f in self.labeled_frames:
            pts = f.get('points', {})
            if old in pts:
                pts[new] = pts.pop(old)
        if old in self.points:
            self.points[new] = self.points.pop(old)
        self._rebuild_keypoint_meta()
        self._save_keypoint_names()
        self._populate_keypoint_buttons()
        self.select_keypoint(new)
        if self.cap:
            self.show_frame(self.current_frame)
        self.status_var.set(f"✏️ 已重命名“{old}”→“{new}”")

    def _populate_keypoint_buttons(self):
        """(Re)build the inline keypoint rows: select ◉ + editable name + ✕,
        plus an ➕ Add Keypoint button. Lives in self.kp_container."""
        for child in self.kp_container.winfo_children():
            child.destroy()
        self.kp_buttons = {}

        for i, name in enumerate(self.keypoint_names):
            row = ttk.Frame(self.kp_container)
            row.pack(fill=tk.X, pady=2)

            rb = ttk.Radiobutton(row, variable=self.kp_select_var, value=name,
                                 command=lambda n=name: self.select_keypoint(n))
            rb.pack(side=tk.LEFT)

            color_canvas = tk.Canvas(row, width=16, height=16, bg='gray',
                                     highlightthickness=1)
            color_canvas.pack(side=tk.LEFT, padx=(0, 3))

            name_var = tk.StringVar(value=name)
            entry = ttk.Entry(row, textvariable=name_var, width=13)
            entry.pack(side=tk.LEFT)
            # Commit rename on Enter / focus-out (deferred so the widget isn't
            # destroyed while still handling its own event)
            entry.bind('<Return>', lambda e, o=name, v=name_var:
                       self.root.after_idle(lambda: self._commit_rename(o, v.get())))
            entry.bind('<FocusOut>', lambda e, o=name, v=name_var:
                       self.root.after_idle(lambda: self._commit_rename(o, v.get())))

            status_label = ttk.Label(row, text="⬜", foreground='gray', width=2)
            status_label.pack(side=tk.LEFT, padx=2)

            ttk.Button(row, text="✕", width=2,
                       command=lambda n=name: self._remove_keypoint(n)).pack(side=tk.RIGHT)

            self.kp_buttons[name] = {
                'radio': rb, 'name_var': name_var, 'entry': entry,
                'color': color_canvas, 'status': status_label, 'placed': False
            }

        ttk.Button(self.kp_container, text="➕ 添加关键点",
                   command=self._add_keypoint).pack(anchor=tk.W, pady=(4, 0))

        if not self.keypoint_names:
            ttk.Label(self.kp_container, text="暂无标签，请点击“➕ 添加关键点”。",
                      foreground='gray', font=('Arial', 8)).pack(anchor=tk.W, pady=2)

        # Rebuild shortcut hints
        for child in self.kp_shortcut_container.winfo_children():
            child.destroy()
        for i, name in enumerate(self.keypoint_names[:9]):
            ttk.Label(self.kp_shortcut_container, text=f"{i+1}: {name[:8]}",
                      foreground='gray', font=('Arial', 8)).pack(side=tk.LEFT, padx=3)

    # *************************************************************
    # *** CORE SMOOTHING FUNCTIONALITY ***
    # *************************************************************
    def apply_moving_average(self):
        """Applies a moving average filter to all stored keypoint coordinates."""
        if not self.labeled_frames:
            return 0

        # We will modify the 'points' dictionary in place
        smoothed_count = 0
        
        for labeled in self.labeled_frames:
            frame = labeled['frame']
            original_points = labeled['points']
            new_points = {}
            
            for kp_name, (x, y) in original_points.items():
                # Collect coordinates from the last N frames for this specific keypoint
                coordinates = []
                start_idx = max(0, self.labeled_frames.index(labeled) - self.smoothing_window_size)
                end_idx = min(len(self.labeled_frames), self.labeled_frames.index(labeled) + 1)

                for i in range(start_idx, end_idx):
                    # Check if the point exists in the windowed frame
                    temp_label = self.labeled_frames[i]
                    if kp_name in temp_label['points']:
                        coords = temp_label['points'][kp_name]
                        coordinates.append((coords[0], coords[1]))

                # Calculate the average (the smoothed point)
                if coordinates:
                    avg_x = int(sum(c[0] for c in coordinates) / len(coordinates))
                    avg_y = int(sum(c[1] for c in coordinates) / len(coordinates))
                    new_points[kp_name] = (avg_x, avg_y)
                else:
                    # Should not happen if the keypoint existed originally
                    new_points[kp_name] = (x, y) 

            labeled['points'] = new_points
            smoothed_count += 1
        
        return smoothed_count

    # ============ YOLO MODEL LOADING ============
    
    def load_yolo_model(self):
        """Load the optional pose-assist model, with progress feedback.

        A missing ultralytics is a normal state rather than an error — keypoints
        can always be placed by hand, and every caller already guards on
        `self.yolo_model is None`. So say so quietly instead of raising a dialog
        the user cannot act on if they deliberately keep AGPL out of the env.
        """
        if YOLO is None:
            self.status_var.set(
                "ℹ️ 姿态辅助已关闭（未安装 ultralytics），请手动放置关键点"
            )
            self.yolo_model = None
            return
        try:
            self.status_var.set("🔄 正在加载 YOLO 模型…")
            self.root.update()
            
            # Use YOLOv8n-pose for keypoint detection
            model_name = "yolov8n-pose.pt"  # Pose estimation model
            
            # Check if model exists locally, if not download
            model_path = Path(model_name)
            if not model_path.exists():
                self.status_var.set(f"📥 正在下载 {model_name}…（仅首次需要）")
                self.root.update()
            
            self.yolo_model = YOLO(model_name)
            self.status_var.set(f"✅ YOLO 姿态模型已加载：{model_name}")
            
        except Exception as e:
            self.status_var.set(f"❌ YOLO 加载失败：{str(e)}")
            messagebox.showerror("YOLO 错误", 
                f"无法加载 YOLO 模型。\n\n错误：{str(e)}\n\n请确认已安装 ultralytics：\npip install ultralytics")
            self.yolo_model = None
    
    # ============ YOLO TRACKING METHODS ============
    
    def track_with_yolo(self, start_frame=None, end_frame=None, use_manual_anchors=True):
        """Enhanced tracking using YOLO pose estimation with prediction fallback."""
        if not self.cap:
            messagebox.showwarning("未加载视频", "请先加载视频")
            return False
        
        if self.yolo_model is None:
            messagebox.showwarning("未加载 YOLO 模型", "YOLO 模型尚未加载，请检查安装情况。")
            return False
        
        # Determine tracking range
        if start_frame is None:
            start_frame = self.current_frame
        
        if end_frame is None or end_frame >= self.total_frames:
            end_frame = self.total_frames - 1
        
        # Collect manual frames for reference
        manual_frames = {}
        for labeled in self.labeled_frames:
            if labeled.get('manual', False):
                frame = labeled['frame']
                if start_frame <= frame <= end_frame:
                    manual_frames[frame] = labeled['points']
        
        # Determine which labels to track
        if hasattr(self, 'manual_labels_only') and self.manual_labels_only.get():
            manual_labels = set()
            for frame_points in manual_frames.values():
                manual_labels.update(frame_points.keys())
            if not manual_labels:
                messagebox.showwarning("没有手动标签", 
                    "请先至少手动标注一帧。")
                return False
        else:
            manual_labels = set(self.keypoint_names)  # Track all
        
        # Reset tracking state for the new run
        self.tracking_active = True
        self.track_history = defaultdict(list) 
        
        self.status_var.set(f"🔄 YOLO 正在跟踪第 {start_frame}-{end_frame} 帧…（按 ESC 停止）")
        self.root.update()
        
        tracked_frames = []
        
        # Process video
        for frame_idx in range(start_frame, end_frame + 1):
            if not self.tracking_active:
                break
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if not ret:
                break
            
            current_frame_points = {}
            temp_results = []

        # 1. Use manual points if available (Highest priority)
        if frame_idx in manual_frames:
            current_frame_points = manual_frames[frame_idx]
            temp_results = [{'points': current_frame_points, 'tracked': True}] 
        else:
            # Run YOLO pose estimation
            results = None
            try:
                results = self.yolo_model.track(
                    frame, 
                    persist=True,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    verbose=False
                )
            except Exception as e:
                print(f"YOLO 在第 {frame_idx} 帧出错：{e}")
                results = None

                # Handle detection results
                if results and results[0].keypoints is not None:
                    keypoints_data = results[0].keypoints
                    
                    for i, (keypoints, box) in enumerate(zip(keypoints_data.xy, results[0].boxes.xyxy)):
                        temp_points = self._map_coco_to_keypoints(keypoints, frame.shape)
                        if not temp_points: continue

                        filtered_points = {}
                        for label in manual_labels:
                            if label in temp_points:
                                filtered_points[label] = temp_points[label]
                        
                        if filtered_points:
                            current_frame_points = filtered_points
                            temp_results.append({
                                'points': current_frame_points, 
                                'tracked': True, 
                                'track_id': int(results[0].boxes.id[i].item()) if results[0].boxes.id is not None and i < len(results[0].boxes.id) else None,
                                'conf': float(results[0].boxes.conf[i].item()) if results[0].boxes.conf is not None and i < len(results[0].boxes.conf) else 0.0
                            })
                    
                    if temp_results:
                        current_frame_points = temp_results[0]['points']

                # Handle detection failure -> use prediction fallback
                else:
                    print(f"⚠️ 警告：YOLO 在第 {frame_idx} 帧丢失跟踪，改用历史数据。")
                    current_frame_points = self._predict_keypoints(start_frame, end_frame)


            # 2. Add the result to tracked_frames and update history/progress
            if current_frame_points:
                tracked_frames.append({
                    'frame': frame_idx,
                    'timestamp': frame_idx / self.fps,
                    'points': current_frame_points.copy(), 
                    'manual': False,
                    # We use a general check to see if *any* detection happened (even bad ones)
                    'tracked': bool(results and results[0].keypoints is not None), 
                    'track_id': temp_results[0].get('track_id') if temp_results else None,
                    'tracking_method': self.tracking_method,
                    'yolo_conf': 1.0 # Placeholder for prediction/manual frames
                })

                # Update history with the successfully calculated points
                for label in manual_labels:
                    if label in current_frame_points:
                        self.track_history[label].append((frame_idx, current_frame_points[label][0], current_frame_points[label][1]))
            
            # Update progress (Only update status if we processed a frame)
            if frame_idx % 20 == 0 and not (start_frame <= frame_idx < end_frame):
                progress = int((frame_idx - start_frame) / (end_frame - start_frame + 1) * 100)
                self.status_var.set(f"🔄 跟踪进度：{progress}%（第 {frame_idx}/{end_frame} 帧）")
                self.root.update()
        
        # Update overall history and label list
        self.labeled_frames = [f for f in self.labeled_frames if f.get('manual', False) or f.get('manual_correction', False)]
        self.labeled_frames.extend(tracked_frames)
        self.labeled_frames.sort(key=lambda x: x['frame'])
        
        # === APPLY SMOOTHING AFTER ALL FRAMES ARE POPULATED (FIXED LINE CALL) ===
        smoothed_count = self.apply_moving_average() 
        print(f"✨ 已对 {smoothed_count} 个点应用平滑处理。") # User feedback
        
        self.progress_bar['value'] = len(self.labeled_frames)

        # Update UI
        self.progress_bar['value'] = len(self.labeled_frames)
        self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} 帧")
        
        if self.video_path:
            self.save_frame_progress()
        
        self.tracking_active = False
        self.status_var.set(f"✅ YOLO 跟踪完成！新增 {len(tracked_frames)} 帧（已应用预测和平滑）")
        
        # Display current frame
        self.show_frame(self.current_frame)
        return True
    
    def _predict_keypoints(self, start_frame, end_frame):
        """Predict keypoint positions using linear interpolation based on history."""
        predicted_points = {}
        for kp_name in self.keypoint_names:
            history = self.track_history[kp_name]
            if len(history) >= 2:
                # Use the last two known points for prediction
                p1 = history[-2]  # (frame, x, y)
                p2 = history[-1]  # (frame, x, y)
                
                # Simple linear extrapolation/interpolation using p1 and p2 movement vector
                dx = p2[1] - p1[1] # change in X
                dy = p2[2] - p1[2] # change in Y
                
                # Predict for the current frame_idx (which is end_frame + 1)
                current_x = p2[1] + dx * 0.5 # Predict half a step forward
                current_y = p2[2] + dy * 0.5
                predicted_points[kp_name] = (int(current_x), int(current_y))
            else:
                # If not enough history, return None or last known point if available
                if self.track_history[kp_name]:
                    last_frame, x, y = self.track_history[kp_name][-1]
                    predicted_points[kp_name] = (x, y)
        return predicted_points

    def get_manual_labels(self):
        """Get the set of labels that have been manually saved"""
        manual_labels = set()
        for labeled in self.labeled_frames:
            if labeled.get('manual', False):
                manual_labels.update(labeled['points'].keys())
        return manual_labels

    def update_label_selection_from_manual(self):
        """Update the UI to only show labels that exist in manual frames"""
        manual_labels = self.get_manual_labels()
        
        # Hide or disable labels not in manual set
        for name in self.keypoint_names:
            if name not in manual_labels:
                # You could disable these buttons or hide them
                self.kp_buttons[name]['radio'].config(state='disabled')
            else:
                self.kp_buttons[name]['radio'].config(state='normal')

    def _map_coco_to_keypoints(self, keypoints, frame_shape):
        """Legacy COCO-pose mapping. Disabled: the pretrained pose model only
        detects generic human-skeleton points, not arbitrary user-defined
        keypoints, so it cannot meaningfully map to them. Use optical flow
        (Track Between Manual) instead. Returns no points."""
        return {}
    
    def _map_to_keypoint(self, class_id, cx, cy, existing_points):
        """Map YOLO class to your keypoint names - simpler fallback"""
        # This is a simplified mapping for when pose estimation fails
        available = [k for k in self.keypoint_names if k not in existing_points]
        if available:
            # Simple assignment in order
            return available[0] if available else None
        return None
    
    def track_with_manual_assistance(self):
        """Interactive tracking: User labels some frames, YOLO tracks the rest"""
        manual_frames = [f for f in self.labeled_frames if f.get('manual', False)]
        
        if len(manual_frames) < 2:
            messagebox.showwarning("需要手动标注帧", 
                "请至少手动标注 2 帧作为参考")
            return False
        
        # Sort manual frames
        manual_frames.sort(key=lambda x: x['frame'])
        
        # Track between manual frames
        total_tracked = 0
        for i in range(len(manual_frames) - 1):
            start_frame = manual_frames[i]['frame']
            end_frame = manual_frames[i + 1]['frame']
            
            if end_frame - start_frame > 1:
                self.status_var.set(f"🔄 正在跟踪片段 {i+1}/{len(manual_frames)-1}：第 {start_frame}-{end_frame} 帧")
                self.root.update()
                
                # Set the current frame to start frame for tracking
                self.current_frame = start_frame
                success = self.track_with_yolo(
                    start_frame=start_frame + 1,
                    end_frame=end_frame - 1,
                    use_manual_anchors=False
                )
                if success:
                    total_tracked += (end_frame - start_frame - 1)
        
        self.status_var.set(f"✅ 手动辅助跟踪完成！新增 {total_tracked} 帧")
        return True
    
    def hybrid_track(self):
        """Hybrid approach: YOLO + manual verification points"""
        if not self.cap:
            messagebox.showwarning("未加载视频", "请先加载视频")
            return
        
        # 1. Find all manual frames
        manual_frames = [f for f in self.labeled_frames if f.get('manual', False)]
        
        if len(manual_frames) < 2:
            messagebox.showwarning("需要手动标注帧", 
                "请至少手动标注 2 帧")
            return
        
        # 2. Remove existing tracked frames
        self.labeled_frames = [f for f in self.labeled_frames if f.get('manual', False)]
        
        # 3. Run YOLO tracking between manual frames
        manual_frames.sort(key=lambda x: x['frame'])
        
        total_tracked = 0
        for i in range(len(manual_frames) - 1):
            start = manual_frames[i]
            end = manual_frames[i + 1]
            
            if end['frame'] - start['frame'] <= 1:
                continue
            
            # Set current frame to start
            self.current_frame = start['frame']
            
            # Track segment
            self.status_var.set(f"🔄 正在跟踪片段：第 {start['frame']}-{end['frame']} 帧")
            self.root.update()
            
            success = self.track_with_yolo(
                start_frame=start['frame'] + 1,
                end_frame=end['frame'] - 1,
                use_manual_anchors=False
            )
            
            if success:
                total_tracked += (end['frame'] - start['frame'] - 1)
        
        # 4. Final interpolation for any gaps
        self.smart_interpolate()
        
        self.status_var.set("✅ 混合跟踪完成！手动标注 + YOLO + 插值")
        self.show_frame(self.current_frame)
    
        # === APPLY SMOOTHING AFTER ALL FRAMES ARE POPULATED ===
        smoothed_count = self.apply_moving_average() 
        self.status_var.set(f"✅ 混合跟踪完成！手动标注 + YOLO + 插值 + 平滑（已平滑 {smoothed_count} 个点）")


    # ============ SPACEBAR HANDLING ============
    
    def on_space_press(self, event=None):
        """Handle spacebar press - toggle playback only once"""
        if not self.space_held:
            self.space_held = True
            self.toggle_play()
        return "break"

    def on_space_release(self, event=None):
        """Handle spacebar release - reset the held flag"""
        self.space_held = False
        return "break"
    
    # ============ PATH NORMALIZATION ============
    
    def normalize_path(self, path):
        return str(Path(path).resolve())
    
    # ============ PROGRESS TRACKING ============
    
    def load_progress(self):
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    self.progress_data = json.load(f)
                self.status_var.set(f"📊 已加载进度：已跟踪 {len(self.progress_data)} 个视频")
            except:
                self.progress_data = {}
                self.status_var.set("未找到有效的进度文件")
        else:
            self.progress_data = {}
            self.status_var.set("未找到进度文件，将从头开始")
    
    def save_progress(self):
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress_data, f, indent=2)
        except Exception as e:
            self.status_var.set(f"⚠️ 无法保存进度：{e}")
    
    def save_frame_progress(self):
        if not self.video_path:
            return
        
        video_key = self.normalize_path(self.video_path)
        self.progress_data[video_key] = {
            'completed': False,
            'frames_labeled': len(self.labeled_frames),
            'total_frames': self.total_frames,
            'last_frame': self.current_frame,
            'last_updated': datetime.now().isoformat(),
            'video_name': os.path.basename(self.video_path),
            'labeled_frames': self.labeled_frames
        }
        self.save_progress()
        self.update_video_progress()
    
    def load_saved_progress(self, video_path):
        video_key = self.normalize_path(video_path)
        if video_key in self.progress_data:
            data = self.progress_data[video_key]
            if 'labeled_frames' in data:
                self.labeled_frames = data['labeled_frames']
                self.progress_bar['value'] = len(self.labeled_frames)
                self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} 帧")
                self.status_var.set(f"🔄 已恢复 {len(self.labeled_frames)} 个已标注帧")
            return data
        return None
    
    def mark_video_complete(self, video_path):
        video_key = self.normalize_path(video_path)
        self.progress_data[video_key] = {
            'completed': True,
            'frames_labeled': len(self.labeled_frames),
            'total_frames': self.total_frames,
            'last_frame': self.current_frame,
            'completion_date': datetime.now().isoformat(),
            'video_name': os.path.basename(video_path),
            'labeled_frames': self.labeled_frames
        }
        self.save_progress()
        self.update_video_progress()

    def get_video_status(self, video_path):
        video_key = self.normalize_path(video_path)
        return self.progress_data.get(video_key)

    def update_video_progress(self):
        if self.video_files:
            total = len(self.video_files)
            completed = 0
            for v in self.video_files:
                status = self.get_video_status(v)
                if status and status.get('completed', False):
                    completed += 1
            
            self.video_progress_label.config(text=f"📊 已完成 {completed}/{total} 个视频")
            self.root.title(f"🎬 视频标注器 - 已完成 {completed}/{total} 个视频")
        else:
            if self.video_path:
                self.video_progress_label.config(text="📊 单视频模式")
                self.root.title(f"🎬 视频标注器 - {os.path.basename(self.video_path)}")
            else:
                self.video_progress_label.config(text="📊 未加载视频")
                self.root.title("🎬 视频标注器 - AI 辅助跟踪")
    
    # ============ UI SETUP ============
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left: Video display
        video_frame = ttk.LabelFrame(main_frame, text="视频播放器", padding=5)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.video_panel = tk.Canvas(video_frame, bg='black')
        self.video_panel.pack(fill=tk.BOTH, expand=True)
        self.video_panel.bind("<Button-1>", self.on_canvas_click)
        self.video_panel.bind("<MouseWheel>", self.on_mouse_wheel)
        self.video_panel.bind("<ButtonPress-2>", self.on_pan_start)
        self.video_panel.bind("<B2-Motion>", self.on_pan_move)
        self.video_panel.bind("<Button-1>", lambda e: self.video_panel.focus_set(), add="+")
        
        # Video controls
        controls = ttk.Frame(video_frame)
        controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(controls, text="⏮", command=self.prev_frame, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="⏪", command=self.prev_10, width=3).pack(side=tk.LEFT, padx=2)
        self.play_btn = ttk.Button(controls, text="▶ 播放", command=self.toggle_play, width=6)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="⏩", command=self.next_10, width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls, text="⏭", command=self.next_frame, width=3).pack(side=tk.LEFT, padx=2)
        
        self.slider = ttk.Scale(controls, from_=0, to=100, orient=tk.HORIZONTAL)
        self.slider.bind("<ButtonRelease-1>", self.slider_changed)
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        self.frame_label = ttk.Label(controls, text="0 / 0")
        self.frame_label.pack(side=tk.RIGHT, padx=5)
        
        # Navigation controls
        nav_frame = ttk.Frame(video_frame)
        nav_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(nav_frame, text="⏮ 上一个视频", 
                  command=self.prev_video, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="⏭ 下一个视频", 
                  command=self.next_video, width=15).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="📊 查看进度", 
                  command=self.show_progress, width=15).pack(side=tk.LEFT, padx=2)
        
        self.video_progress_label = ttk.Label(nav_frame, text="📊 已完成 0/0 个视频", foreground='blue')
        self.video_progress_label.pack(side=tk.RIGHT, padx=5)

        # Right panel (scrollable so controls never get clipped on small windows)
        right_outer = ttk.LabelFrame(main_frame, text="控制面板", padding=2)
        right_outer.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        right_canvas = tk.Canvas(right_outer, borderwidth=0, highlightthickness=0, width=240)
        right_vsb = ttk.Scrollbar(right_outer, orient=tk.VERTICAL, command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_vsb.set)
        right_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_frame = ttk.Frame(right_canvas, padding=8)
        self._right_window = right_canvas.create_window((0, 0), window=right_frame, anchor='nw')
        self.right_canvas = right_canvas
        self.right_frame_inner = right_frame

        right_frame.bind("<Configure>",
                         lambda e: right_canvas.configure(scrollregion=right_canvas.bbox("all")))
        right_canvas.bind("<Configure>",
                          lambda e: right_canvas.itemconfig(self._right_window, width=e.width))

        def _on_right_wheel(event):
            right_canvas.yview_scroll(int(-event.delta / 120), "units")
        right_canvas.bind("<Enter>", lambda e: right_canvas.bind_all("<MouseWheel>", _on_right_wheel))
        right_canvas.bind("<Leave>", lambda e: right_canvas.unbind_all("<MouseWheel>"))
        
        # Current video indicator — always visible, never overwritten by action logs
        self.current_video_var = tk.StringVar(value="未加载视频")
        ttk.Label(right_frame, textvariable=self.current_video_var,
                  font=('Arial', 9, 'bold'), foreground='#888888',
                  wraplength=160, justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 4))

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # File controls
        ttk.Label(right_frame, text="📁 文件", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        ttk.Button(right_frame, text="打开视频", command=self.open_video, width=15).pack(pady=2)
        ttk.Button(right_frame, text="打开文件夹", command=self.open_folder, width=15).pack(pady=2)

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Keypoint section (select to place, edit name inline, add/remove)
        ttk.Label(right_frame, text="🎯 关键点",
                  font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(right_frame, text="点击 ◉ 选择 · 编辑名称 · ✕ 删除",
                  font=('Arial', 8), foreground='gray').pack(anchor=tk.W)

        # Shared selection variable + rebuildable container for the keypoint rows
        self.kp_select_var = tk.StringVar(value="")
        self.kp_container = ttk.Frame(right_frame)
        self.kp_container.pack(fill=tk.X)
        self.kp_buttons = {}

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Quick select shortcuts (also rebuilt when keypoints change)
        ttk.Label(right_frame, text="⌨️ 快捷键：", font=('Arial', 9)).pack(anchor=tk.W)
        self.kp_shortcut_container = ttk.Frame(right_frame)
        self.kp_shortcut_container.pack(fill=tk.X, pady=2)

        # Populate keypoint rows + shortcut hints
        self._populate_keypoint_buttons()

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Label actions
        ttk.Label(right_frame, text="📝 操作", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))

        ttk.Button(right_frame, text="✅ 保存当前帧", command=self.save_frame, width=15).pack(pady=2)
        ttk.Button(right_frame, text="🚫 标记遮挡 (H)", command=self.mark_occluded, width=15).pack(pady=2)
        ttk.Button(right_frame, text="↩️ 撤销上一步", command=self.undo_point, width=15).pack(pady=2)
        ttk.Button(right_frame, text="🗑️ 清空标注", command=self.clear_frame, width=15).pack(pady=2)
        ttk.Button(right_frame, text="⏭️ 跳过当前帧", command=self.skip_frame, width=15).pack(pady=2)
        ttk.Button(right_frame, text="✅ 标记完成", command=self.mark_current_complete, width=15).pack(pady=2)
        ttk.Button(right_frame, text="💾 导出标签", command=self.export_labels, width=15).pack(pady=2)

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Optical Flow tracking (recommended for custom keypoints)
        ttk.Label(right_frame, text="🌊 光流跟踪（推荐）",
                  font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 2))
        ttk.Label(right_frame,
                  text="跟随真实像素填充手动标注帧之间的画面。\n会准确衔接每个手动标注帧。",
                  font=('Arial', 8), foreground='gray', justify=tk.LEFT).pack(anchor=tk.W)
        ttk.Button(right_frame, text="🌊 跟踪手动帧之间 (O)",
                   command=self.optical_flow_track, width=22).pack(pady=4)

        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # YOLO Tracking controls
        ttk.Label(right_frame, text="🤖 YOLO 跟踪", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        ttk.Label(right_frame,
                  text="⚠️ 预训练姿态模型无法识别自定义关键点，\n可能发生漂移。建议仅在训练自己的模型后使用。",
                  font=('Arial', 8), foreground='#aa5500', justify=tk.LEFT).pack(anchor=tk.W)
        
        tracker_frame = ttk.Frame(right_frame)
        tracker_frame.pack(fill=tk.X, pady=2)
        
        # Add checkbox for manual labels only
        self.manual_labels_only = tk.BooleanVar(value=True)
        ttk.Checkbutton(tracker_frame, text="🎯 仅跟踪手动标签", 
                        variable=self.manual_labels_only).pack(anchor=tk.W)

        self.tracker_var = tk.StringVar(value="botsort")
        ttk.Radiobutton(tracker_frame, text="BoTSORT（均衡）", 
                       variable=self.tracker_var, value="botsort").pack(anchor=tk.W)
        ttk.Radiobutton(tracker_frame, text="ByteTrack（快速）", 
                       variable=self.tracker_var, value="bytetrack").pack(anchor=tk.W)
        
        # YOLO confidence threshold
        conf_frame = ttk.Frame(right_frame)
        conf_frame.pack(fill=tk.X, pady=2)
        ttk.Label(conf_frame, text="置信度：").pack(side=tk.LEFT)
        self.conf_scale = ttk.Scale(conf_frame, from_=0.1, to=0.9, orient=tk.HORIZONTAL, length=80)
        self.conf_scale.set(0.5)
        self.conf_scale.pack(side=tk.LEFT, padx=5)
        self.conf_label = ttk.Label(conf_frame, text="0.5")
        self.conf_label.pack(side=tk.LEFT)
        self.conf_scale.configure(command=lambda v: self.conf_label.config(text=f"{float(v):.1f}"))
        
        ttk.Button(right_frame, text="🎯 YOLO 跟踪当前段", 
                  command=self.track_with_yolo, width=15).pack(pady=2)
        ttk.Button(right_frame, text="🔄 混合跟踪（手动+YOLO）", 
                  command=self.hybrid_track, width=15).pack(pady=2)
        ttk.Button(right_frame, text="📊 跟踪全部帧", 
                  command=lambda: self.track_with_yolo(0, self.total_frames-1), width=15).pack(pady=2)
        ttk.Button(right_frame, text="🧹 清除自动跟踪", 
                  command=self.clear_tracked_frames, width=15).pack(pady=2)
        
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Backup interpolation
        ttk.Label(right_frame, text="🔄 备用插值", font=('Arial', 9)).pack(anchor=tk.W)
        ttk.Button(right_frame, text="📐 线性插值", 
                  command=self.linear_interpolate_all, width=15).pack(pady=1)
        
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Adjustment controls
        ttk.Label(right_frame, text="🎯 调整关键点", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        
        adjust_frame = ttk.Frame(right_frame)
        adjust_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(adjust_frame, text="◀", command=lambda: self.adjust_point(-5, 0), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(adjust_frame, text="▲", command=lambda: self.adjust_point(0, -5), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(adjust_frame, text="▼", command=lambda: self.adjust_point(0, 5), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(adjust_frame, text="▶", command=lambda: self.adjust_point(5, 0), width=3).pack(side=tk.LEFT, padx=1)
        
        adjust_frame2 = ttk.Frame(right_frame)
        adjust_frame2.pack(fill=tk.X, pady=2)
        
        ttk.Button(adjust_frame2, text="微调 ◀", command=lambda: self.adjust_point(-1, 0), width=5).pack(side=tk.LEFT, padx=1)
        ttk.Button(adjust_frame2, text="微调 ▲", command=lambda: self.adjust_point(0, -1), width=5).pack(side=tk.LEFT, padx=1)
        ttk.Button(adjust_frame2, text="微调 ▼", command=lambda: self.adjust_point(0, 1), width=5).pack(side=tk.LEFT, padx=1)
        ttk.Button(adjust_frame2, text="微调 ▶", command=lambda: self.adjust_point(1, 0), width=5).pack(side=tk.LEFT, padx=1)
        
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Progress
        ttk.Label(right_frame, text="📊 进度", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0,5))
        self.progress_bar = ttk.Progressbar(right_frame, length=150, mode='determinate')
        self.progress_bar.pack(pady=5)
        self.progress_label = ttk.Label(right_frame, text="0 / 0 帧")
        self.progress_label.pack()
        
        # Status bar
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
        
        # Select first keypoint
        if self.keypoint_names:
            self.select_keypoint(self.keypoint_names[0])
        
        self.video_panel.focus_set()

    def _finalize_window(self):
        """Force a clean initial layout so the window opens at the right size
        instead of needing a manual resize."""
        self.root.update_idletasks()
        # Size the scrollable control panel to its natural width so nothing clips
        try:
            req_w = self.right_frame_inner.winfo_reqwidth()
            self.right_canvas.configure(width=req_w)
        except Exception:
            pass
        self.root.update_idletasks()
        # Fit the window to its content, clamped to the screen, and centered
        req_w = self.root.winfo_reqwidth()
        req_h = self.root.winfo_reqheight()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        win_w = min(max(req_w, 1100), screen_w - 80)
        win_h = min(max(req_h, 750), screen_h - 80)
        x = max((screen_w - win_w) // 2, 0)
        y = max((screen_h - win_h) // 3, 0)
        self.root.geometry(f"{win_w}x{win_h}+{x}+{y}")
        self.root.minsize(1000, 650)

    def bind_shortcuts(self):
        """Bind keyboard shortcuts globally"""
        def _g(fn):
            """Wrap fn so it is ignored when an Entry/Text widget has focus."""
            def _wrapped(event):
                w = self.root.focus_get()
                if isinstance(w, (tk.Entry, tk.Text, ttk.Entry)):
                    return
                return fn(event)
            return _wrapped

        self.root.bind_all('<KeyPress-space>', _g(self.on_space_press))
        self.root.bind_all('<KeyRelease-space>', _g(self.on_space_release))
        self.root.bind_all('<Right>', _g(self.next_frame))
        self.root.bind_all('<Left>', _g(self.prev_frame))
        self.root.bind_all('<Control-s>', self.save_frame)
        self.root.bind_all('<Control-z>', self.undo_point)
        self.root.bind_all('<Escape>', _g(self.skip_frame))
        self.root.bind_all('<h>', _g(self.mark_occluded))
        self.root.bind_all('<H>', _g(self.mark_occluded))
        for i in range(9):  # support up to 9 keypoints via number keys
            self.root.bind_all(f'<Key-{i+1}>', _g(lambda e, idx=i: self.select_keypoint_by_index(idx)))
        self.root.bind_all('<t>', _g(lambda e: self.track_with_yolo()))
        self.root.bind_all('<o>', _g(lambda e: self.optical_flow_track()))
    
    # ============ KEYPOINT SELECTION ============
    
    def select_keypoint(self, name):
        if name not in self.keypoint_names:
            return
            
        self.current_kp = self.keypoint_names.index(name)
        self.kp_select_var.set(name)

        for kp_name, data in self.kp_buttons.items():
            if kp_name == name:
                data['color'].configure(bg='yellow')
                self.status_var.set(f"已选择：{self.keypoint_display_names[name]} - 请在视频画面中点击放置")
            else:
                if data['placed']:
                    data['color'].configure(bg='green')
                else:
                    data['color'].configure(bg='gray')
        
        if self.cap:
            self.slider_update = False
            self.show_frame(self.current_frame, update_slider=False)
            self.slider_update = True
    
    def select_keypoint_by_index(self, idx, event=None):
        if idx < len(self.keypoint_names):
            self.select_keypoint(self.keypoint_names[idx])
        return "break"
    
    # ============ VIDEO LOADING ============
    
    def open_video(self):
        file_path = filedialog.askopenfilename(
            title="选择视频",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if file_path:
            self.video_files = []
            self.current_video_idx = 0
            self.load_video(file_path)
            self.status_var.set(f"📹 单视频模式：{os.path.basename(file_path)}")
    
    def open_folder(self):
        folder = filedialog.askdirectory(title="选择包含视频的文件夹")
        if folder:
            videos = set()
            for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV', '*.MKV']:
                videos.update(Path(folder).glob(ext))
            
            if videos:
                self.video_files = sorted(videos, key=natural_sort_key)
                self.current_video_idx = 0
                self.status_var.set(f"📁 文件夹中找到 {len(videos)} 个视频")
                self.update_video_progress()
                self.find_first_uncompleted()
            else:
                messagebox.showwarning("没有视频", "该文件夹中未找到视频文件")
                self.video_files = []
                self.current_video_idx = 0
                self.update_video_progress()
    
    def find_first_uncompleted(self):
        if not self.video_files:
            return
        
        for i, video_path in enumerate(self.video_files):
            status = self.get_video_status(video_path)
            if not status or not status.get('completed', False):
                self.current_video_idx = i
                self.load_video(str(video_path))
                self.status_var.set(f"📹 从此视频开始：{os.path.basename(video_path)}")
                return
        
        self.current_video_idx = 0
        self.load_video(str(self.video_files[0]))
        messagebox.showinfo("全部完成！", "所有视频都已完成！\n将重新加载第一个视频。")
        self.status_var.set("🎉 所有视频均已完成！")
    
    def load_video(self, path):
        self.video_path = path
        status = self.get_video_status(path)
        self.current_video_exported = bool(status and status.get('completed', False))
        if hasattr(self, 'current_video_var'):
            self.current_video_var.set(f"🎬 {os.path.basename(path)}")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("错误", f"无法打开视频：\n{path}")
            return
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.current_frame = 0
        self.points = {}
        self.occluded = set()
        self.labeled_frames = []
        self.tracking_data = {}
        self.tracking_active = False
        
        for name in self.keypoint_names:
            self.kp_buttons[name]['placed'] = False
            self.kp_buttons[name]['color'].configure(bg='gray')
            self.kp_buttons[name]['status'].configure(text='⬜', foreground='gray')
        
        if self.keypoint_names:
            self.select_keypoint(self.keypoint_names[0])
        
        self.slider.config(to=self.total_frames-1)
        self.frame_label.config(text=f"0 / {self.total_frames}")
        self.progress_bar['maximum'] = self.total_frames
        
        saved_data = self.load_saved_progress(path)
        
        if saved_data:
            if saved_data.get('completed', False):
                self.status_var.set(f"✅ 已完成：{os.path.basename(path)}")
                self.progress_bar['value'] = self.total_frames
                self.progress_label.config(text=f"{self.total_frames} / {self.total_frames} 帧")
                self.show_frame(0)
            else:
                last_frame = saved_data.get('last_frame', 0)
                if last_frame > 0 and last_frame < self.total_frames:
                    self.current_frame = last_frame
                    self.status_var.set(f"🔄 从第 {last_frame} 帧继续")
                    self.show_frame(last_frame)
                else:
                    self.show_frame(0)
        else:
            self.show_frame(0)
            self.progress_bar['value'] = 0
            self.progress_label.config(text=f"0 / {self.total_frames} 帧")
        
        self.status_var.set(f"📹 已加载：{os.path.basename(path)}（{self.total_frames} 帧）")
        self.update_video_progress()
        
        if self.video_files:
            idx = self.current_video_idx + 1
            total = len(self.video_files)
            self.status_var.set(f"📹 视频 {idx}/{total}：{os.path.basename(path)}（{self.total_frames} 帧）")
        
        self.video_panel.focus_set()
        self.show_frame(0, update_slider=True)
    
    # ============ VIDEO DISPLAY ============
    
    def on_canvas_click(self, event):
        if not self.cap:
            return

        if not self.keypoint_names or self.current_kp >= len(self.keypoint_names):
            self.status_var.set("⚠️ 尚未定义关键点，请先在 🎯 关键点面板中点击“➕ 添加关键点”。")
            return

        name = self.keypoint_names[self.current_kp]
        x, y = event.x, event.y

        if name not in self.points:
            self.points[name] = []
        self.points[name].append((x, y))
        self.occluded.discard(name)
        self.kp_buttons[name]['placed'] = True
        self.kp_buttons[name]['color'].configure(bg='green')
        self.kp_buttons[name]['status'].configure(text='✅', foreground='green')

        self.status_var.set(
            f"✅ 已放置：{self.keypoint_display_names[name]}，位置 ({x}, {y})")

        self.advance_to_next_keypoint()
        
        self.slider_update = False
        self.show_frame(self.current_frame)
        self.slider_update = True
    
    def advance_to_next_keypoint(self):
        def done(name):
            return bool(self.points.get(name)) or name in self.occluded

        if all(done(name) for name in self.keypoint_names):
            self.status_var.set("所有关键点均已放置或标记遮挡！请点击“保存当前帧”或按 Ctrl+S")
            return

        for name in self.keypoint_names:
            if not done(name):
                self.select_keypoint(name)
                break

    def mark_occluded(self, event=None):
        """Mark the selected keypoint as hidden/inside for this frame (visibility 0)
        instead of placing a position. Interpolation/optical-flow skip it, so no
        bounding box is invented while the part is occluded."""
        if not self.keypoint_names or self.current_kp >= len(self.keypoint_names):
            return "break"
        name = self.keypoint_names[self.current_kp]
        self.points.pop(name, None)          # remove any placed position
        self.occluded.add(name)
        self.kp_buttons[name]['placed'] = False
        self.kp_buttons[name]['color'].configure(bg='#663333')
        self.kp_buttons[name]['status'].configure(text='🚫', foreground='#cc7777')
        self.status_var.set(f"🚫 {name}：遮挡/位于内部（当前帧不绘制框）")
        self.advance_to_next_keypoint()
        self.slider_update = False
        self.show_frame(self.current_frame)
        self.slider_update = True
        return "break"
    
    def show_frame(self, frame_idx, update_slider=True):
        if not self.cap:
            return
        
        if frame_idx < 0:
            frame_idx = 0
        elif frame_idx >= self.total_frames:
            frame_idx = self.total_frames - 1
        
        self.current_frame = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        
        h, w = frame.shape[:2]
        display_h = 600
        scale = display_h / h
        display_w = int(w * scale)
        frame = cv2.resize(frame, (display_w, display_h))
        
        # Draw current points being placed
        for name, instances in self.points.items():
            color = self.colors[name]
            label_base = name.replace('_', ' ').title()
            for i, coords in enumerate(instances):
                cv2.circle(frame, coords, 14, color, 2)
                cv2.circle(frame, coords, 10, color, -1)
                cv2.circle(frame, coords, 12, (255, 255, 255), 1)
                lbl = f"{label_base} {i+1}" if len(instances) > 1 else label_base
                cv2.putText(frame, lbl, (coords[0]+15, coords[1]-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw saved frames (manual + tracked)
        for labeled in self.labeled_frames:
            if labeled['frame'] == frame_idx:
                for name, raw in labeled['points'].items():
                    color = self.colors.get(name, (255, 255, 255))
                    instances = _pts(raw)
                    is_tracked = labeled.get('tracked', False)
                    label_base = name.replace('_', ' ').title()
                    for i, coords in enumerate(instances):
                        lbl = f"{label_base} {i+1}" if len(instances) > 1 else label_base
                        if is_tracked:
                            cv2.circle(frame, coords, 12, color, 2)
                            cv2.circle(frame, coords, 6, color, -1)
                            cv2.putText(frame, f"{lbl} (T)",
                                        (coords[0]+15, coords[1]+15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
                        else:
                            cv2.circle(frame, coords, 14, color, 2)
                            cv2.circle(frame, coords, 10, color, -1)
                            cv2.circle(frame, coords, 12, (255, 255, 255), 1)
                            cv2.putText(frame, lbl, (coords[0]+15, coords[1]-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Show currently selected keypoint (skip the prompt if it's occluded)
        if self.current_kp < len(self.keypoint_names):
            name = self.keypoint_names[self.current_kp]
            if not self.kp_buttons[name]['placed'] and name not in self.occluded:
                cv2.line(frame, (display_w//2 - 30, display_h//2), 
                        (display_w//2 + 30, display_h//2), self.colors[name], 2)
                cv2.line(frame, (display_w//2, display_h//2 - 30), 
                        (display_w//2, display_h//2 + 30), self.colors[name], 2)
                cv2.putText(frame, f"Click to place: {self.keypoint_display_names[name]}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors[name], 2)
        
        # Show frame info
        manual_count = sum(1 for f in self.labeled_frames if f.get('manual', False))
        tracked_count = sum(1 for f in self.labeled_frames if f.get('tracked', False))
        cv2.putText(frame, f"Frame: {frame_idx}/{self.total_frames} | Manual: {manual_count} Tracked: {tracked_count}", 
                (display_w-450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show tracker status
        if self.tracking_active:
            cv2.putText(frame, "🔴 YOLO TRACKING ACTIVE", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Convert to PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_panel.imgtk = imgtk
        self.video_panel.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.video_panel.config(width=display_w, height=display_h)
        
        if update_slider:
            self.slider.set(frame_idx)
        self.frame_label.config(text=f"{frame_idx} / {self.total_frames}")
    
    def on_mouse_wheel(self, event):
        pass
    
    def on_pan_start(self, event):
        self.pan_x = event.x
        self.pan_y = event.y
    
    def on_pan_move(self, event):
        pass
    
    # ============ VIDEO CONTROLS ============
    
    def toggle_play(self, event=None):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="▶ 播放")
            if hasattr(self, '_after_id'):
                try:
                    self.root.after_cancel(self._after_id)
                except:
                    pass
        else:
            self.is_playing = True
            self.play_btn.config(text="⏸ 暂停")
            self.play_video()
        return "break"
    
    def play_video(self):
        if not self.is_playing or not self.cap:
            return
        
        next_frame = self.current_frame + 1
        if next_frame >= self.total_frames:
            self.is_playing = False
            self.play_btn.config(text="▶ 播放")
            return
        
        self.show_frame(next_frame, update_slider=True)
        self._after_id = self.root.after(1000 // self.fps, self.play_video)

    def next_frame(self, event=None):
        if not self.cap:
            return "break"
        
        if self.current_frame < self.total_frames - 1:
            self.show_frame(self.current_frame + 1, update_slider=True)
        else:
            if self.is_playing:
                self.stop_playback()
                self.status_var.set("🎬 已到达视频末尾")
        return "break"
    
    def stop_playback(self):
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="▶ 播放")
            if hasattr(self, '_after_id'):
                try:
                    self.root.after_cancel(self._after_id)
                except:
                    pass

    def prev_frame(self, event=None):
        if self.cap and self.current_frame > 0:
            self.show_frame(self.current_frame - 1, update_slider=True)
        return "break"

    def next_10(self):
        if self.cap and self.current_frame < self.total_frames - 10:
            self.show_frame(self.current_frame + 10, update_slider=True)

    def prev_10(self):
        if self.cap and self.current_frame > 10:
            self.show_frame(self.current_frame - 10, update_slider=True)

    def slider_changed(self, event):
        if not self.cap:
            return
        
        try:
            frame = int(float(self.slider.get()))
            if frame != self.current_frame:
                self.show_frame(frame, update_slider=False)
        except:
            pass
    
    # ============ LABELING ACTIONS ============
    
    def save_frame(self, event=None):
        if not self.cap:
            return "break"
        
        if not self.points and not self.occluded:
            # Check if there are tracked points for this frame
            existing_frame = None
            for labeled in self.labeled_frames:
                if labeled['frame'] == self.current_frame:
                    existing_frame = labeled
                    break

            if existing_frame:
                self.status_var.set(f"ℹ️ 第 {self.current_frame} 帧已有 {len(existing_frame['points'])} 个点")
                return "break"
            else:
                messagebox.showwarning("没有关键点", "保存前请至少放置一个关键点，或将一个关键点标记为遮挡。")
                return "break"

        # Save the frame with points (+ any occluded keypoints, exported as visibility 0)
        frame_data = {
            'frame': self.current_frame,
            'timestamp': self.current_frame / self.fps,
            'points': self.points.copy(),
            'occluded': sorted(self.occluded),
            'manual': True,
            'tracked': False
        }
        
        # Check if frame already exists, replace if so
        existing_idx = None
        for i, labeled in enumerate(self.labeled_frames):
            if labeled['frame'] == self.current_frame:
                existing_idx = i
                break
        
        if existing_idx is not None:
            self.labeled_frames[existing_idx] = frame_data
            self.status_var.set(f"🔄 已更新第 {self.current_frame} 帧")
        else:
            self.labeled_frames.append(frame_data)
            self.status_var.set(f"✅ 第 {self.current_frame} 帧已保存！")
        
        self.current_video_exported = False
        self.progress_bar['value'] = len(self.labeled_frames)
        self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} 帧")
        
        if self.video_path:
            self.save_frame_progress()
        
        # Clear points
        self.points = {}
        self.occluded = set()
        for name in self.keypoint_names:
            self.kp_buttons[name]['placed'] = False
            self.kp_buttons[name]['color'].configure(bg='gray')
            self.kp_buttons[name]['status'].configure(text='⬜', foreground='gray')

        if self.keypoint_names:
            self.select_keypoint(self.keypoint_names[0])

        if self.is_playing:
            self.next_frame()
        else:
            self.slider_update = False
            self.show_frame(self.current_frame, update_slider=True)
            self.slider_update = True
        
        self.video_panel.focus_set()
        return "break"
    
    def undo_point(self, event=None):
        if self.points:
            last = list(self.points.keys())[-1]
            self.points[last].pop()
            count = len(self.points[last])
            if count == 0:
                del self.points[last]
                self.kp_buttons[last]['placed'] = False
                self.kp_buttons[last]['color'].configure(bg='gray')
                self.kp_buttons[last]['status'].configure(text='⬜', foreground='gray')
            else:
                self.kp_buttons[last]['status'].configure(
                    text=f'✅×{count}' if count > 1 else '✅', foreground='green')
            self.select_keypoint(last)
            self.status_var.set(f"↩️ 已撤销：删除 {last} 的一个实例（剩余 {count} 个）")
            self.slider_update = False
            self.show_frame(self.current_frame)
            self.slider_update = True
        return "break"

    def clear_frame(self):
        if self.labeled_frames:
            if not messagebox.askyesno("清空全部标注帧", 
                                    f"确定删除全部 {len(self.labeled_frames)} 个已标注帧吗？"):
                return
        
        self.points = {}
        self.occluded = set()
        self.labeled_frames = []
        for name in self.keypoint_names:
            self.kp_buttons[name]['placed'] = False
            self.kp_buttons[name]['color'].configure(bg='gray')
            self.kp_buttons[name]['status'].configure(text='⬜', foreground='gray')

        self.clear_saved_progress()
        
        if self.keypoint_names:
            self.select_keypoint(self.keypoint_names[0])
        
        self.status_var.set("🗑️ 已清空全部标签")
        self.current_video_exported = False
        self.progress_bar['value'] = 0
        self.progress_label.config(text=f"0 / {self.total_frames} 帧")
        
        self.slider_update = False
        self.show_frame(self.current_frame)
        self.slider_update = True
    
    def clear_saved_progress(self):
        if not self.video_path:
            return
        
        video_key = self.normalize_path(self.video_path)
        if video_key in self.progress_data:
            del self.progress_data[video_key]
            self.save_progress()
            self.status_var.set(f"🗑️ 已清除 {os.path.basename(self.video_path)} 的进度")
            self.update_video_progress()

    def skip_frame(self, event=None):
        self.points = {}
        self.occluded = set()
        for name in self.keypoint_names:
            self.kp_buttons[name]['placed'] = False
            self.kp_buttons[name]['color'].configure(bg='gray')
            self.kp_buttons[name]['status'].configure(text='⬜', foreground='gray')
        
        if self.keypoint_names:
            self.select_keypoint(self.keypoint_names[0])
        
        self.status_var.set(f"⏭️ 已跳过第 {self.current_frame} 帧")
        
        if self.is_playing:
            self.next_frame()
        else:
            self.slider_update = False
            self.show_frame(self.current_frame, update_slider=True)
            self.slider_update = True
        
        return "break"

    def mark_current_complete(self):
        if not self.video_path:
            messagebox.showwarning("未加载视频", "当前未加载视频")
            return
        
        if not self.labeled_frames:
            if not messagebox.askyesno("没有标签", "仍要标记为完成吗？"):
                return
        
        if self.labeled_frames and not self.current_video_exported:
            if messagebox.askyesno("先导出？", "标记为完成前是否先导出标签？"):
                exported = self.export_labels()
                if not exported:
                    return
        
        self.mark_video_complete(self.video_path)
        self.status_var.set(f"✅ 已将 {os.path.basename(self.video_path)} 标记为完成")
        self.stop_playback()
        
        if self.video_files and len(self.video_files) > 1:
            if messagebox.askyesno("下一个视频", "是否移动到下一个视频？"):
                self._advance_to_next_video_force()
    
    # ============ INTERPOLATION METHODS ============
    
    def smart_interpolate(self):
        """Hybrid interpolation using manual frames + tracked data"""
        if len(self.labeled_frames) < 2:
            messagebox.showwarning("至少需要 2 个已标注帧")
            return
        
        # Sort by frame number
        self.labeled_frames.sort(key=lambda x: x['frame'])
        
        # Remove old interpolated frames
        self.labeled_frames = [f for f in self.labeled_frames if not f.get('interpolated', False)]
        
        # Get manual frames as anchors
        anchors = [f for f in self.labeled_frames if f.get('manual', False)]
        
        if len(anchors) < 2:
            messagebox.showwarning("插值至少需要 2 个手动标注帧")
            return
        
        new_frames = []
        
        for i in range(len(anchors) - 1):
            start = anchors[i]
            end = anchors[i + 1]
            
            # Distance between frames
            distance = end['frame'] - start['frame']
            
            if distance <= 1:
                continue
            
            # Interpolate between them
            for frame_num in range(start['frame'] + 1, end['frame']):
                t = (frame_num - start['frame']) / distance
                points = {}
                
                for kp_name in self.keypoint_names:
                    if kp_name in start['points'] and kp_name in end['points']:
                        p1 = start['points'][kp_name]
                        p2 = end['points'][kp_name]
                        points[kp_name] = (
                            int(p1[0] + t * (p2[0] - p1[0])),
                            int(p1[1] + t * (p2[1] - p1[1]))
                        )
                    elif kp_name in start['points']:
                        points[kp_name] = start['points'][kp_name]
                    elif kp_name in end['points']:
                        points[kp_name] = end['points'][kp_name]
                
                if points:
                    new_frames.append({
                        'frame': frame_num,
                        'timestamp': frame_num / self.fps,
                        'points': points,
                        'manual': False,
                        'tracked': False,
                        'interpolated': True
                    })
        
        # Add interpolated frames
        self.labeled_frames.extend(new_frames)
        self.labeled_frames.sort(key=lambda x: x['frame'])
        
        # Update UI
        self.progress_bar['value'] = len(self.labeled_frames)
        self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} 帧")
        self.status_var.set(f"✅ 已插值 {len(new_frames)} 帧")
    
    def linear_interpolate_all(self):
        """Backup: Linear interpolation between manual frames"""
        manual_frames = [f for f in self.labeled_frames if f.get('manual', False)]
        
        if len(manual_frames) < 2:
            messagebox.showwarning("标签不足", "至少需要 2 个手动标注帧")
            return False
        
        # Remove existing interpolated frames
        self.labeled_frames = [f for f in self.labeled_frames if f.get('manual', False)]
        
        sorted_labels = sorted(manual_frames, key=lambda x: x['frame'])
        frames_added = 0
        
        for i in range(len(sorted_labels) - 1):
            start = sorted_labels[i]
            end = sorted_labels[i + 1]
            
            if end['frame'] - start['frame'] <= 1:
                continue
            
            # Don't interpolate keypoints marked occluded in either anchor
            skip = set(start.get('occluded', [])) | set(end.get('occluded', []))

            for frame in range(start['frame'] + 1, end['frame']):
                t = (frame - start['frame']) / (end['frame'] - start['frame'])
                interpolated_points = {}

                for kp_name in self.keypoint_names:
                    if kp_name in skip:
                        continue
                    if kp_name in start['points'] and kp_name in end['points']:
                        ia = _pts(start['points'][kp_name])
                        ib = _pts(end['points'][kp_name])
                        interp = [(int(p1[0] + t * (p2[0] - p1[0])),
                                   int(p1[1] + t * (p2[1] - p1[1])))
                                  for p1, p2 in zip(ia, ib)]
                        interp += ia[len(ib):]
                        interp += ib[len(ia):]
                        interpolated_points[kp_name] = interp
                    elif kp_name in start['points']:
                        interpolated_points[kp_name] = _pts(start['points'][kp_name])
                    elif kp_name in end['points']:
                        interpolated_points[kp_name] = _pts(end['points'][kp_name])
                
                if interpolated_points:
                    self.labeled_frames.append({
                        'frame': frame,
                        'timestamp': frame / self.fps,
                        'points': interpolated_points,
                        'manual': False,
                        'tracked': False,
                        'interpolated': True
                    })
                    frames_added += 1
        
        self.labeled_frames = sorted(self.labeled_frames, key=lambda x: x['frame'])
        self.progress_bar['value'] = len(self.labeled_frames)
        self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} 帧")
        
        if self.video_path:
            self.save_frame_progress()
        
        self.status_var.set(f"✅ 已插值 {frames_added} 帧")
        return True
    
    # ============ OPTICAL FLOW TRACKING (recommended for custom keypoints) ============

    def _get_video_dims(self):
        """Return native (width, height) of the loaded video in pixels."""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def optical_flow_track(self):
        """Propagate manual labels to the frames in between using bidirectional
        Lucas-Kanade optical flow, blended between each pair of manual anchors.

        Unlike pre-trained YOLO pose, this follows the actual pixels you clicked,
        so it works for arbitrary custom keypoints and lands exactly on both
        manual anchors (no accumulated drift between them)."""
        if not self.cap:
            messagebox.showwarning("未加载视频", "请先加载视频")
            return False

        manual = sorted([f for f in self.labeled_frames if f.get('manual', False)],
                        key=lambda x: x['frame'])
        if len(manual) < 2:
            messagebox.showwarning("需要手动标注帧",
                "请先至少手动标注 2 帧。\n"
                "光流会填充两个手动标注帧之间的画面。")
            return False

        vid_w, vid_h = self._get_video_dims()
        # Labels are stored in DISPLAY space (video is resized to 600px tall in
        # show_frame). Optical flow must run in native video pixels, so convert
        # in/out using this scale: display = video * scale.
        disp_h = 600
        scale = disp_h / vid_h if vid_h else 1.0

        # Drop previously generated (non-manual) frames so re-running is clean
        self.labeled_frames = [f for f in self.labeled_frames
                               if f.get('manual', False) or f.get('manual_correction', False)]

        MAX_SEGMENT = 400  # beyond this, optical flow gets unreliable -> linear
        new_frames = []
        total_segments = len(manual) - 1
        self.tracking_active = True

        for si in range(total_segments):
            if not self.tracking_active:
                break
            a, b = manual[si], manual[si + 1]
            fa, fb = a['frame'], b['frame']
            if fb - fa <= 1:
                continue

            self.status_var.set(f"🌊 光流跟踪：片段 {si+1}/{total_segments} "
                                f"（第 {fa}-{fb} 帧）")
            self.root.update()

            # Fully skip only keypoints occluded in BOTH anchors. If occluded at
            # just one end, it's still tracked from the visible end and dropped on
            # loss — so the visible approach keeps a box and it vanishes when gone.
            skip = set(a.get('occluded', [])) & set(b.get('occluded', []))

            if fb - fa > MAX_SEGMENT:
                seg = self._linear_segment(fa, fb, a['points'], b['points'], skip)
                method = 'linear_far'
            else:
                seg = self._optical_flow_segment(fa, fb, a['points'], b['points'],
                                                 scale, vid_w, vid_h, skip)
                method = 'optical_flow'

            for fi, pts in seg.items():
                new_frames.append({
                    'frame': fi,
                    'timestamp': fi / self.fps,
                    'points': pts,
                    'manual': False,
                    'tracked': True,
                    'interpolated': True,
                    'tracking_method': method,
                })

        self.tracking_active = False
        self.labeled_frames.extend(new_frames)
        self.labeled_frames.sort(key=lambda x: x['frame'])

        self.progress_bar['value'] = len(self.labeled_frames)
        self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} 帧")
        if self.video_path:
            self.save_frame_progress()

        self.status_var.set(f"✅ 光流跟踪完成：已填充 {len(new_frames)} 帧，"
                            f"位于 {len(manual)} 个手动锚点之间")
        self.slider_update = False
        self.show_frame(self.current_frame)
        self.slider_update = True
        return True

    def _linear_segment(self, fa, fb, A, B, skip=None):
        """Plain linear interpolation between two anchors (display-space points).
        Keypoints in `skip` (occluded in an anchor) are omitted -> visibility 0."""
        skip = skip or set()
        out = {}
        span = fb - fa
        for fi in range(fa + 1, fb):
            t = (fi - fa) / span
            pts = {}
            for k in self.keypoint_names:
                if k in skip:
                    continue
                # Only interpolate a point present at BOTH anchors. If it's at just
                # one end it's entering/leaving -> drop it (label disappears) rather
                # than carrying a stale position across the gap.
                if k in A and k in B:
                    ia, ib = _pts(A[k]), _pts(B[k])
                    interp = [(int(round(p1[0] + t * (p2[0] - p1[0]))),
                               int(round(p1[1] + t * (p2[1] - p1[1]))))
                              for p1, p2 in zip(ia, ib)]
                    interp += ia[len(ib):]  # carry extra instances from A
                    interp += ib[len(ia):]  # carry extra instances from B
                    pts[k] = interp
            if pts:
                out[fi] = pts
        return out

    def _optical_flow_segment(self, fa, fb, A, B, scale, vid_w, vid_h, skip=None):
        """Bidirectional LK optical flow between anchors A (frame fa) and B (frame fb).
        Points come in/out in DISPLAY space; tracking runs in native video pixels.
        Keypoints in `skip` (occluded in an anchor) are omitted -> visibility 0.
        Returns {frame_idx: {kp: (x, y)}} for the intermediate frames."""
        skip = skip or set()
        # Buffer grayscale frames for the whole segment (sequential read = fast)
        grays = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, fa)
        for _ in range(fa, fb + 1):
            ret, fr = self.cap.read()
            if not ret:
                break
            grays.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        if len(grays) < (fb - fa + 1):
            return self._linear_segment(fa, fb, A, B, skip)  # couldn't read full segment

        n = fb - fa  # number of steps
        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

        def to_video(pt):
            return [pt[0] / scale, pt[1] / scale]

        # Forward/backward must agree within this many native pixels, else the
        # point is treated as occluded/drifted and dropped.
        fb_thresh = max(20.0, 0.02 * max(vid_w, vid_h))

        # Expand multi-instance: "cls" → "cls__0", "cls__1" etc.
        def _expand(pts_dict):
            out = {}
            for k, val in pts_dict.items():
                for i, pt in enumerate(_pts(val)):
                    out[f"{k}__{i}"] = pt
            return out

        def _collapse(pts_dict):
            out = {}
            for key, pt in pts_dict.items():
                name = key.rsplit('__', 1)[0] if '__' in key else key
                out.setdefault(name, []).append(pt)
            return out

        A_exp = _expand({k: v for k, v in A.items() if k not in skip})
        B_exp = _expand({k: v for k, v in B.items() if k not in skip})

        kps_f = list(A_exp.keys())
        kps_b = list(B_exp.keys())

        # --- Forward track from anchor A ---
        fwd, fwd_ok = {}, {}
        if kps_f:
            p = np.array([to_video(A_exp[k]) for k in kps_f], dtype=np.float32).reshape(-1, 1, 2)
            ok = [True] * len(kps_f)
            for i in range(1, n + 1):
                p2, st, err = cv2.calcOpticalFlowPyrLK(grays[i - 1], grays[i], p, None, **lk)
                fp, fo = {}, {}
                for j, k in enumerate(kps_f):
                    x, y = float(p2[j, 0, 0]), float(p2[j, 0, 1])
                    good = (ok[j] and st[j, 0] == 1 and err[j, 0] < 30.0
                            and 0 <= x < vid_w and 0 <= y < vid_h)
                    ok[j] = good
                    fp[k], fo[k] = (x, y), good
                fwd[fa + i], fwd_ok[fa + i] = fp, fo
                p = p2

        # --- Backward track from anchor B ---
        bwd, bwd_ok = {}, {}
        if kps_b:
            p = np.array([to_video(B_exp[k]) for k in kps_b], dtype=np.float32).reshape(-1, 1, 2)
            ok = [True] * len(kps_b)
            for i in range(1, n + 1):
                idx = n - i
                p2, st, err = cv2.calcOpticalFlowPyrLK(grays[idx + 1], grays[idx], p, None, **lk)
                bp, bo = {}, {}
                for j, k in enumerate(kps_b):
                    x, y = float(p2[j, 0, 0]), float(p2[j, 0, 1])
                    good = (ok[j] and st[j, 0] == 1 and err[j, 0] < 30.0
                            and 0 <= x < vid_w and 0 <= y < vid_h)
                    ok[j] = good
                    bp[k], bo[k] = (x, y), good
                bwd[fa + idx], bwd_ok[fa + idx] = bp, bo
                p = p2

        # Collect all expanded keys present in either anchor
        all_exp_keys = set(kps_f) | set(kps_b)

        # --- Blend forward + backward, weighted by position between anchors ---
        out = {}
        for fi in range(fa + 1, fb):
            t = (fi - fa) / float(n)
            pts_exp = {}
            for k in all_exp_keys:
                f_ok = fwd_ok.get(fi, {}).get(k, False)
                b_ok = bwd_ok.get(fi, {}).get(k, False)
                if f_ok and b_ok:
                    fx, fy = fwd[fi][k]; bx, by = bwd[fi][k]
                    if (fx - bx) ** 2 + (fy - by) ** 2 > fb_thresh * fb_thresh:
                        continue
                    vx, vy = (1 - t) * fx + t * bx, (1 - t) * fy + t * by
                elif f_ok:
                    vx, vy = fwd[fi][k]
                elif b_ok:
                    vx, vy = bwd[fi][k]
                else:
                    continue
                pts_exp[k] = (int(round(vx * scale)), int(round(vy * scale)))
            pts = _collapse(pts_exp)
            if pts:
                out[fi] = pts
        return out

    def clear_tracked_frames(self):
        """Remove all tracked frames from labeled_frames"""
        tracked_count = sum(1 for f in self.labeled_frames if f.get('tracked', False))
        
        if tracked_count == 0:
            messagebox.showinfo("没有跟踪帧", "没有可清除的自动跟踪帧")
            return
        
        if messagebox.askyesno("清除自动跟踪", f"确定删除全部 {tracked_count} 个自动跟踪帧吗？"):
            self.labeled_frames = [f for f in self.labeled_frames if not f.get('tracked', False)]
            
            self.progress_bar['value'] = len(self.labeled_frames)
            self.progress_label.config(text=f"{len(self.labeled_frames)} / {self.total_frames} 帧")
            
            if self.video_path:
                self.save_frame_progress()
            
            self.status_var.set(f"🧹 已删除 {tracked_count} 个自动跟踪帧")
            
            self.slider_update = False
            self.show_frame(self.current_frame)
            self.slider_update = True
    
    def adjust_point(self, dx, dy):
        """Adjust the selected point by dx, dy pixels"""
        if not self.points:
            # Check if current frame has points
            current_frame_points = None
            for labeled in self.labeled_frames:
                if labeled['frame'] == self.current_frame:
                    current_frame_points = labeled['points']
                    break
            
            if current_frame_points:
                # Adjust the current keypoint in the frame
                name = self.keypoint_names[self.current_kp]
                if name in current_frame_points:
                    x, y = current_frame_points[name]
                    new_point = (max(0, x + dx), max(0, y + dy))
                    
                    # Update in labeled_frames
                    for labeled in self.labeled_frames:
                        if labeled['frame'] == self.current_frame:
                            labeled['points'][name] = new_point
                            break
                    
                    self.status_var.set(f"🔧 已将 {name} 调整到 {new_point}")
                    
                    # Update display
                    self.slider_update = False
                    self.show_frame(self.current_frame)
                    self.slider_update = True
                    
                    # Mark as not exported
                    self.current_video_exported = False
                    
                    if self.video_path:
                        self.save_frame_progress()
                else:
                    self.status_var.set(f"⚠️ 当前帧未找到 {name}")
            else:
                self.status_var.set("⚠️ 当前帧没有关键点")
        else:
            # Adjust points being placed
            name = self.keypoint_names[self.current_kp]
            if self.points.get(name):
                x, y = self.points[name][-1]
                self.points[name][-1] = (max(0, x + dx), max(0, y + dy))
                self.status_var.set(f"🔧 已将 {name} 的第 {len(self.points[name])} 个实例调整到 {self.points[name][-1]}")
                
                self.slider_update = False
                self.show_frame(self.current_frame)
                self.slider_update = True
    
    # ============ NAVIGATION ============
    
    def _advance_to_next_video_force(self):
        if not self.video_files:
            return
        
        self.stop_playback()
        
        for i in range(self.current_video_idx + 1, len(self.video_files)):
            video_path = self.video_files[i]
            status = self.get_video_status(video_path)
            if not status or not status.get('completed', False):
                self.current_video_idx = i
                self.load_video(str(video_path))
                self.status_var.set(f"⏭ 已移动到：{os.path.basename(video_path)}")
                return
        
        for i in range(0, self.current_video_idx + 1):
            video_path = self.video_files[i]
            status = self.get_video_status(video_path)
            if not status or not status.get('completed', False):
                self.current_video_idx = i
                self.load_video(str(video_path))
                self.status_var.set(f"⏭ 已循环到：{os.path.basename(video_path)}")
                return
        
        self.current_video_idx = 0
        self.load_video(str(self.video_files[0]))
        messagebox.showinfo("🎉 全部完成！", "所有视频均已完成！")
        self.status_var.set("🎉 所有视频均已完成！")
    
    def next_video(self):
        if not self.video_files:
            messagebox.showinfo("没有视频", "请先打开一个文件夹")
            return
        
        if self.labeled_frames and not self.current_video_exported:
            status = self.get_video_status(self.video_path)
            if status and status.get('completed', False):
                self._advance_to_next_video_force()
                return
                
            if not messagebox.askyesno("先导出标签", 
                                    f"当前有 {len(self.labeled_frames)} 个已标注帧。\n移动前是否先导出？"):
                self._advance_to_next_video_force()
                return
            else:
                exported = self.export_labels()
                if not exported:
                    return
        
        self._advance_to_next_video_force()

    def prev_video(self):
        if not self.video_files:
            messagebox.showinfo("没有视频", "请先打开一个文件夹")
            return
        
        if self.labeled_frames and not self.current_video_exported:
            status = self.get_video_status(self.video_path)
            if status and status.get('completed', False):
                self._advance_to_prev_video_force()
                return
                
            if not messagebox.askyesno("先导出标签", 
                                    f"当前有 {len(self.labeled_frames)} 个已标注帧。\n移动前是否先导出？"):
                self._advance_to_prev_video_force()
                return
            else:
                exported = self.export_labels()
                if exported:
                    self._advance_to_prev_video_force()
                return
        
        self._advance_to_prev_video_force()

    def _advance_to_prev_video_force(self):
        if not self.video_files:
            return
        
        self.stop_playback()
        
        if self.labeled_frames and self.video_path:
            self.save_frame_progress()
        
        if self.current_video_idx > 0:
            self.current_video_idx -= 1
            self.load_video(str(self.video_files[self.current_video_idx]))
            self.status_var.set(f"⏮ 已移动到：{os.path.basename(self.video_files[self.current_video_idx])}")
        else:
            messagebox.showinfo("已到文件夹开头", "当前已经是第一个视频！")
    
    def show_progress(self):
        if not self.video_files:
            messagebox.showinfo("没有视频", "请先打开一个文件夹")
            return
        
        progress_window = tk.Toplevel(self.root)
        progress_window.title("📊 标注进度")
        progress_window.geometry("700x500")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        scrollbar = tk.Scrollbar(progress_window)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(progress_window, yscrollcommand=scrollbar.set, font=('Courier', 10))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.config(command=listbox.yview)
        
        listbox.insert(tk.END, f"{'状态':<8} {'视频名称':<50} {'帧数':<15} {'日期':<20}")
        listbox.insert(tk.END, "=" * 95)
        
        completed_count = 0
        for video_path in self.video_files:
            video_name = os.path.basename(video_path)
            status = self.get_video_status(video_path)
            
            if status and status.get('completed', False):
                frames = status.get('frames_labeled', 0)
                total = status.get('total_frames', 0)
                date = status.get('completion_date', '')[:10]
                listbox.insert(tk.END, f"{'✅ 完成':<8} {video_name:<50} {frames}/{total:<10} {date:<20}")
                completed_count += 1
            elif status and not status.get('completed', False):
                frames = status.get('frames_labeled', 0)
                total = status.get('total_frames', 0)
                listbox.insert(tk.END, f"{'🔄 进行中':<8} {video_name:<50} {frames}/{total:<10} {'':<20}")
            else:
                listbox.insert(tk.END, f"{'⬜ 待处理':<8} {video_name:<50} {'0/0':<15} {'':<20}")
        
        total = len(self.video_files)
        percent = (completed_count/total*100) if total > 0 else 0
        
        listbox.insert(tk.END, "=" * 95)
        listbox.insert(tk.END, f"📊 已完成：{completed_count}/{total} 个视频（{percent:.1f}%）")
        
        btn_frame = ttk.Frame(progress_window)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="关闭", command=progress_window.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🔄 重置进度", 
                  command=lambda: self.reset_progress(progress_window)).pack(side=tk.LEFT, padx=5)
        
        progress_window.bind('<Escape>', lambda e: progress_window.destroy())
        self.video_panel.focus_set()
    
    def reset_progress(self, window):
        if messagebox.askyesno("重置进度", "确定重置全部进度吗？"):
            self.progress_data = {}
            self.save_progress()
            self.update_video_progress()
            window.destroy()
            self.status_var.set("🔄 进度已重置")
            messagebox.showinfo("重置完成", "全部进度已重置。")
    
    # ============ EXPORT ============
    
    def export_labels(self):
        if not self.labeled_frames:
            messagebox.showwarning("没有标签", "当前没有已标注帧")
            return False
        
        video_name = os.path.splitext(os.path.basename(self.video_path))[0] if self.video_path else "labels"
        default_filename = f"{video_name}_labels.json"

        # Auto-save to labels/<action_subfolder>/ derived from the video's parent folder.
        # Falls back to a save dialog only when the video path is unknown.
        export_path = None
        if self.video_path:
            video_dir = os.path.dirname(os.path.abspath(self.video_path))
            action_subfolder = os.path.basename(video_dir)
            project_root = os.path.dirname(video_dir)
            save_dir = os.path.join(project_root, "labels", action_subfolder)
            os.makedirs(save_dir, exist_ok=True)
            export_path = os.path.join(save_dir, default_filename)

        if not export_path:
            export_path = filedialog.asksaveasfilename(
                title="导出标签",
                defaultextension=".json",
                initialfile=default_filename,
                filetypes=[("JSON 文件", "*.json")]
            )

        if not export_path:
            return False
        
        # Labels are stored in DISPLAY space (video resized to 600px tall in
        # show_frame). train_yolo.py normalizes by the NATIVE frame size, so we
        # convert display -> native video pixels here. inv_scale = video_h / 600.
        vid_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self.cap else 0
        vid_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self.cap else 0
        disp_h = 600
        inv_scale = (vid_h / disp_h) if vid_h else 1.0

        export_keyframes = []
        for f in self.labeled_frames:
            fc = dict(f)
            # Group by display name (merges source_a + source_a~2 → source_a: [[p1],[p2]])
            grouped: dict = {}
            for k, v in f['points'].items():
                disp = _disp(k)
                scaled = [[int(round(x * inv_scale)), int(round(y * inv_scale))]
                          for x, y in _pts(v)]
                grouped.setdefault(disp, []).extend(scaled)
            fc['points'] = grouped
            export_keyframes.append(fc)

        data = {
            'video': os.path.basename(self.video_path) if self.video_path else "unknown",
            'total_frames': self.total_frames,
            'fps': self.fps,
            'frame_width': vid_w,
            'frame_height': vid_h,
            'coordinate_space': 'video_pixels',
            'keyframes': export_keyframes,
            'keypoint_names': self.keypoint_names,
            'export_date': datetime.now().isoformat(),
            'tracking_method': 'optical_flow + manual',
            'confidence_threshold': self.confidence_threshold
        }
        
        try:
            with open(export_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"[导出] 已保存 → {export_path}")
            self.current_video_exported = True

            if self.video_path:
                self.mark_video_complete(self.video_path)
            self.status_var.set(f"✅ 已导出：{export_path}")
            
            if self.video_files and len(self.video_files) > 1:
                if self.is_playing:
                    self.is_playing = False
                    self.play_btn.config(text="▶ 播放")
                    if hasattr(self, '_after_id'):
                        self.root.after_cancel(self._after_id)
                
                self._advance_to_next_video_force()
            
            return True
                        
        except Exception as e:
            messagebox.showerror("导出失败", f"导出失败：\n{str(e)}")
            self.status_var.set(f"❌ 导出失败：{e}")
            return False

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from modules.system.debug_console import force_utf8_stdio
    force_utf8_stdio()
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("❌ 未安装 PIL/Pillow。请运行：pip install pillow")
        exit(1)
    
    # Optional pose assist (see the import note at the top of this file).
    try:
        import ultralytics
        print(f"✅ 姿态辅助可用（ultralytics {ultralytics.__version__}）")
    except ImportError:
        print("ℹ️ 姿态辅助已关闭——未安装 ultralytics。仍可正常手动标注，"
              关键点将由手动放置。（可选：pip install ultralytics；"
              "AGPL，仅开发环境使用，不随发行版分发。）")
    
    root = tk.Tk()
    app = VideoLabelerGUI(root)
    root.mainloop()