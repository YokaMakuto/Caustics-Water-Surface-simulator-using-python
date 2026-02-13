import numpy as np
from numba import jit, prange
import cv2
import threading
import time
import math
from typing import Tuple, Optional, List
import customtkinter as ctk
from PIL import Image, ImageTk
try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

WAVELENGTH_R = np.float32(650.0)
WAVELENGTH_G = np.float32(550.0)
WAVELENGTH_B = np.float32(450.0)

CAUCHY_A = np.float32(1.3247)
CAUCHY_B = np.float32(3300.0)

MU_R = np.float32(0.20)
MU_G = np.float32(0.05)
MU_B = np.float32(0.01)

UNITS = {
    "wave_scale": "",
    "wave_height": "m",
    "micro_scale": "",
    "micro_height": "m",
    "gerstner_amp": "m",
    "gerstner_wl": "m",
    "gerstner_speed": "m/s",
    "water_depth": "m",
    "exposure": "EV",
    "bloom_threshold": "",
    "bloom_intensity": "",
    "vignette": "",
    "sss_intensity": "",
    "sss_radius": "px",
    "light1_x": "°",
    "light1_y": "°",
    "light1_intensity": "",
    "light2_x": "°",
    "light2_y": "°",
    "light2_intensity": "",
}


@jit(nopython=True, cache=True)
def fast_floor(x):
    return int(np.floor(x))

@jit(nopython=True, cache=True)
def gradient_2d(h, x, y):
    h = h & 7
    if h == 0: return x + y
    elif h == 1: return -x + y
    elif h == 2: return x - y
    elif h == 3: return -x - y
    elif h == 4: return x
    elif h == 5: return -x
    elif h == 6: return y
    else: return -y

@jit(nopython=True, cache=True)
def simplex_noise_2d(x, y, perm):
    F2 = 0.5 * (np.sqrt(3.0) - 1.0)
    G2 = (3.0 - np.sqrt(3.0)) / 6.0
    s = (x + y) * F2
    i = fast_floor(x + s)
    j = fast_floor(y + s)
    t = (i + j) * G2
    X0 = i - t
    Y0 = j - t
    x0 = x - X0
    y0 = y - Y0
    if x0 > y0:
        i1, j1 = 1, 0
    else:
        i1, j1 = 0, 1
    x1 = x0 - i1 + G2
    y1 = y0 - j1 + G2
    x2 = x0 - 1.0 + 2.0 * G2
    y2 = y0 - 1.0 + 2.0 * G2
    ii = i & 255
    jj = j & 255
    n0 = n1 = n2 = 0.0
    t0 = 0.5 - x0*x0 - y0*y0
    if t0 >= 0:
        t0 *= t0
        gi0 = perm[(ii + perm[jj & 255]) & 255]
        n0 = t0 * t0 * gradient_2d(gi0, x0, y0)
    t1 = 0.5 - x1*x1 - y1*y1
    if t1 >= 0:
        t1 *= t1
        gi1 = perm[(ii + i1 + perm[(jj + j1) & 255]) & 255]
        n1 = t1 * t1 * gradient_2d(gi1, x1, y1)
    t2 = 0.5 - x2*x2 - y2*y2
    if t2 >= 0:
        t2 *= t2
        gi2 = perm[(ii + 1 + perm[(jj + 1) & 255]) & 255]
        n2 = t2 * t2 * gradient_2d(gi2, x2, y2)
    return 70.0 * (n0 + n1 + n2)


@jit(nopython=True, cache=True)
def compute_ripple(x, y, cx, cy, t, spawn_time, amplitude, frequency, decay):
    age = t - spawn_time
    if age < 0:
        return 0.0
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    wave_speed = 2.0
    wave_front = wave_speed * age
    if dist > wave_front + 0.5:
        return 0.0
    phase = dist * frequency - age * wave_speed * frequency
    env = amplitude * np.exp(-decay * age) * np.exp(-0.5 * (dist - wave_front)**2)
    return env * np.sin(phase)

@jit(nopython=True, parallel=True, cache=True)
def generate_water_surface_with_ripples(width, height, time, perm,
                                        wave_scale, wave_height, micro_scale, micro_height,
                                        gerstner_amp, gerstner_wl, gerstner_speed,
                                        ripple_centers, ripple_times, ripple_count):
    surface = np.zeros((height, width), dtype=np.float32)
    for j in prange(height):
        for i in range(width):
            u = i / width
            v = j / height
            n1 = simplex_noise_2d(u * wave_scale + time * 0.3, v * wave_scale + time * 0.2, perm)
            n2 = simplex_noise_2d(u * wave_scale * 2.0 + time * 0.5, v * wave_scale * 2.0 - time * 0.3, perm) * 0.5
            n3 = simplex_noise_2d(u * wave_scale * 4.0 - time * 0.4, v * wave_scale * 4.0 + time * 0.6, perm) * 0.25
            simplex_height = (n1 + n2 + n3) * wave_height
            k = 2.0 * np.pi / gerstner_wl
            omega = gerstner_speed * k
            phase = k * (0.707 * u * 10.0 + 0.707 * v * 10.0) - omega * time
            gz = gerstner_amp * np.cos(phase)
            micro = simplex_noise_2d(u * micro_scale + time * 2.0, v * micro_scale - time * 1.5, perm) * micro_height
            ripple_total = 0.0
            for r in range(ripple_count):
                cx = ripple_centers[r, 0]
                cy = ripple_centers[r, 1]
                spawn_t = ripple_times[r]
                ripple_total += compute_ripple(u, v, cx, cy, time, spawn_t, 0.1, 30.0, 0.8)
            surface[j, i] = simplex_height + gz + micro + ripple_total
    return surface


@jit(nopython=True, cache=True)
def cauchy_ior(wavelength_nm):
    return CAUCHY_A + CAUCHY_B / (wavelength_nm * wavelength_nm)

@jit(nopython=True, cache=True)
def beer_lambert_absorption(distance, mu):
    return np.exp(-mu * distance)

@jit(nopython=True, cache=True)
def compute_surface_normal(surface, i, j, width, height, cell_size):
    left = surface[j, max(0, i-1)]
    right = surface[j, min(width-1, i+1)]
    up = surface[max(0, j-1), i]
    down = surface[min(height-1, j+1), i]
    dhdx = (right - left) / (2.0 * cell_size)
    dhdy = (down - up) / (2.0 * cell_size)
    nx = -dhdx
    ny = -dhdy
    nz = 1.0
    length = np.sqrt(nx*nx + ny*ny + nz*nz)
    return nx/length, ny/length, nz/length

@jit(nopython=True, cache=True)
def refract_ray(dx, dy, dz, nx, ny, nz, ior):
    eta = 1.0 / ior
    cos_i = -(dx*nx + dy*ny + dz*nz)
    sin2_t = eta * eta * (1.0 - cos_i * cos_i)
    if sin2_t > 1.0:
        rx = dx + 2.0 * cos_i * nx
        ry = dy + 2.0 * cos_i * ny
        rz = dz + 2.0 * cos_i * nz
        return rx, ry, rz, False
    cos_t = np.sqrt(1.0 - sin2_t)
    rx = eta * dx + (eta * cos_i - cos_t) * nx
    ry = eta * dy + (eta * cos_i - cos_t) * ny
    rz = eta * dz + (eta * cos_i - cos_t) * nz
    return rx, ry, rz, True

@jit(nopython=True, cache=True)
def inverse_refract_ray(dx, dy, dz, nx, ny, nz, ior):
    eta = ior  
    cos_t = -(dx*nx + dy*ny + dz*nz)
    sin2_i = eta * eta * (1.0 - cos_t * cos_t)
    if sin2_i > 1.0:
        return dx, dy, dz, False
    cos_i = np.sqrt(1.0 - sin2_i)
    ix = eta * dx + (eta * cos_t - cos_i) * nx
    iy = eta * dy + (eta * cos_t - cos_i) * ny
    iz = eta * dz + (eta * cos_t - cos_i) * nz
    return ix, iy, iz, True

@jit(nopython=True, parallel=True, cache=True)
def ray_trace_kernel_multilight(surface, caustic_r, caustic_g, caustic_b,
                                width, height, water_depth, cell_size,
                                lights, light_colors, num_lights):
    floor_width = caustic_r.shape[1]
    floor_height = caustic_r.shape[0]
    ior_r = cauchy_ior(WAVELENGTH_R)
    ior_g = cauchy_ior(WAVELENGTH_G)
    ior_b = cauchy_ior(WAVELENGTH_B)
    
    for j in prange(height):
        for i in range(width):
            surface_x = i * cell_size
            surface_y = j * cell_size
            surface_z = surface[j, i]
            nx, ny, nz = compute_surface_normal(surface, i, j, width, height, cell_size)
            
            for light_idx in range(num_lights):
                light_dx = lights[light_idx, 0]
                light_dy = lights[light_idx, 1]
                light_dz = lights[light_idx, 2]
                light_r = light_colors[light_idx, 0]
                light_g = light_colors[light_idx, 1]
                light_b = light_colors[light_idx, 2]
                
                for wavelength_idx in range(3):
                    if wavelength_idx == 0:
                        ior, mu, light_c = ior_r, MU_R, light_r
                    elif wavelength_idx == 1:
                        ior, mu, light_c = ior_g, MU_G, light_g
                    else:
                        ior, mu, light_c = ior_b, MU_B, light_b
                    
                    rx, ry, rz, refracted = refract_ray(light_dx, light_dy, light_dz, nx, ny, nz, ior)
                    if not refracted or rz >= 0:
                        continue
                    
                    t = (-water_depth - surface_z) / rz
                    hit_x = surface_x + rx * t
                    hit_y = surface_y + ry * t
                    travel_dist = t * cell_size
                    absorption = beer_lambert_absorption(travel_dist, mu)
                    
                    floor_u = hit_x / (width * cell_size)
                    floor_v = hit_y / (height * cell_size)
                    fx = floor_u * (floor_width - 1)
                    fy = floor_v * (floor_height - 1)
                    
                    if fx < 0 or fx >= floor_width - 1 or fy < 0 or fy >= floor_height - 1:
                        continue
                    
                    ix = int(fx)
                    iy = int(fy)
                    wx = fx - ix
                    wy = fy - iy
                    w00 = (1 - wx) * (1 - wy)
                    w10 = wx * (1 - wy)
                    w01 = (1 - wx) * wy
                    w11 = wx * wy
                    energy = absorption * 0.5 * light_c
                    
                    if wavelength_idx == 0:
                        caustic_r[iy, ix] += w00 * energy
                        caustic_r[iy, ix+1] += w10 * energy
                        caustic_r[iy+1, ix] += w01 * energy
                        caustic_r[iy+1, ix+1] += w11 * energy
                    elif wavelength_idx == 1:
                        caustic_g[iy, ix] += w00 * energy
                        caustic_g[iy, ix+1] += w10 * energy
                        caustic_g[iy+1, ix] += w01 * energy
                        caustic_g[iy+1, ix+1] += w11 * energy
                    else:
                        caustic_b[iy, ix] += w00 * energy
                        caustic_b[iy, ix+1] += w10 * energy
                        caustic_b[iy+1, ix] += w01 * energy
                        caustic_b[iy+1, ix+1] += w11 * energy

@jit(nopython=True, cache=True)
def find_real_object_position(apparent_x, apparent_y, water_depth, surface_height,
                              observer_height, ior):
    total_height = observer_height + water_depth
    view_dx = apparent_x / total_height
    view_dy = apparent_y / total_height
    view_dz = -1.0
    view_len = np.sqrt(view_dx**2 + view_dy**2 + view_dz**2)
    view_dx /= view_len
    view_dy /= view_len
    view_dz /= view_len
    
    nx, ny, nz = 0.0, 0.0, 1.0
    rx, ry, rz, success = refract_ray(view_dx, view_dy, view_dz, nx, ny, nz, ior)
    
    if not success or rz >= 0:
        return apparent_x, apparent_y, -water_depth, 0.0, 0.0
    
    t = (-water_depth - surface_height) / rz
    real_x = surface_height / (-view_dz) * view_dx + rx * t
    real_y = surface_height / (-view_dz) * view_dy + ry * t
    real_z = -water_depth
    
    displacement_x = real_x - apparent_x
    displacement_y = real_y - apparent_y
    
    return real_x, real_y, real_z, displacement_x, displacement_y


def apply_sss(image, intensity=0.3, radius=5):
    if intensity <= 0:
        return image
    if radius % 2 == 0:
        radius += 1
    blurred = cv2.GaussianBlur(image, (radius, radius), 0)
    blue_scatter = blurred.copy()
    blue_scatter[:, :, 0] *= 1.2
    blue_scatter[:, :, 1] *= 1.1
    return image * (1 - intensity) + blue_scatter * intensity

def apply_bloom(image, threshold=0.8, blur_size=21, intensity=0.5):
    bright = np.maximum(image - threshold, 0)
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.GaussianBlur(bright, (blur_size, blur_size), 0)
    return image + blurred * intensity

def aces_tonemap(x):
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)

def apply_vignette(image, strength=0.3):
    h, w = image.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    vignette = 1 - strength * (dist / max_dist) ** 2
    return image * vignette[:, :, np.newaxis]

def post_process(caustic_r, caustic_g, caustic_b, bloom_threshold, bloom_intensity,
                 vignette_strength, exposure, sss_intensity=0.0, sss_radius=5):
    image = np.stack([caustic_r, caustic_g, caustic_b], axis=-1).astype(np.float32)
    image *= exposure
    if sss_intensity > 0:
        image = apply_sss(image, sss_intensity, int(sss_radius))
    for c in range(3):
        image[:, :, c] = apply_bloom(image[:, :, c], bloom_threshold, 21, bloom_intensity)
    image = aces_tonemap(image)
    image = apply_vignette(image, vignette_strength)
    image = (image * 255).astype(np.uint8)
    return image


class GPUAccelerator:
    
    def __init__(self):
        self.available = False
        self.ctx = None
        self.queue = None
        self.program = None
        
        if not OPENCL_AVAILABLE:
            return
        
        try:
            platforms = cl.get_platforms()
            if not platforms:
                return
            devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
            if not devices:
                devices = platforms[0].get_devices(device_type=cl.device_type.CPU)
            if not devices:
                return
            self.ctx = cl.Context(devices=[devices[0]])
            self.queue = cl.CommandQueue(self.ctx)
            self.device_name = devices[0].name
            self.available = True
        except Exception:
            self.available = False
    
    def get_device_name(self):
        return self.device_name if self.available else "CPU (Numba)"


PRESETS = {
    "Calm Pool": {
        "wave_scale": 3.0, "wave_height": 0.15, "micro_scale": 40.0, "micro_height": 0.02,
        "gerstner_amp": 0.05, "gerstner_wl": 3.0, "gerstner_speed": 0.5, "water_depth": 2.0,
        "bloom_threshold": 0.7, "bloom_intensity": 0.4, "vignette": 0.25, "exposure": 1.5,
        "sss_intensity": 0.2, "sss_radius": 7,
        "light1_x": 5.0, "light1_y": 5.0, "light1_intensity": 1.0,
        "light2_x": -10.0, "light2_y": 10.0, "light2_intensity": 0.0,
    },
    "Stormy Ocean": {
        "wave_scale": 2.0, "wave_height": 0.5, "micro_scale": 60.0, "micro_height": 0.08,
        "gerstner_amp": 0.3, "gerstner_wl": 2.0, "gerstner_speed": 2.0, "water_depth": 4.0,
        "bloom_threshold": 0.6, "bloom_intensity": 0.6, "vignette": 0.4, "exposure": 1.2,
        "sss_intensity": 0.4, "sss_radius": 11,
        "light1_x": 15.0, "light1_y": 15.0, "light1_intensity": 1.2,
        "light2_x": -5.0, "light2_y": 20.0, "light2_intensity": 0.3,
    },
    "Jell-O": {
        "wave_scale": 5.0, "wave_height": 0.3, "micro_scale": 80.0, "micro_height": 0.01,
        "gerstner_amp": 0.15, "gerstner_wl": 1.5, "gerstner_speed": 0.3, "water_depth": 1.5,
        "bloom_threshold": 0.5, "bloom_intensity": 0.7, "vignette": 0.2, "exposure": 2.0,
        "sss_intensity": 0.6, "sss_radius": 15,
        "light1_x": 0.0, "light1_y": 0.0, "light1_intensity": 1.0,
        "light2_x": 0.0, "light2_y": 0.0, "light2_intensity": 0.0,
    },
    "Oil Slick": {
        "wave_scale": 4.0, "wave_height": 0.08, "micro_scale": 100.0, "micro_height": 0.03,
        "gerstner_amp": 0.02, "gerstner_wl": 5.0, "gerstner_speed": 0.2, "water_depth": 0.8,
        "bloom_threshold": 0.4, "bloom_intensity": 0.8, "vignette": 0.35, "exposure": 2.5,
        "sss_intensity": 0.1, "sss_radius": 5,
        "light1_x": 3.0, "light1_y": 3.0, "light1_intensity": 1.0,
        "light2_x": -15.0, "light2_y": 5.0, "light2_intensity": 0.5,
    },
}

class CausticsProApp:
    def __init__(self):
        np.random.seed(42)
        self.perm = np.random.permutation(256).astype(np.int32)
        self.sim_width = 256
        self.sim_height = 256
        self.preview_width = 512
        self.preview_height = 512
        self.cell_size = np.float32(0.05)
        self.time = 0.0
        self.running = True
        self.paused = False
        self.rendering_4k = False
        self.exporting_video = False
        self.params = dict(PRESETS["Calm Pool"])
        
        # Interactive ripples
        self.max_ripples = 10
        self.ripple_centers = np.zeros((self.max_ripples, 2), dtype=np.float32)
        self.ripple_times = np.zeros(self.max_ripples, dtype=np.float32) - 100
        self.ripple_count = 0
        
        # Coin simulation
        self.coin_real_pos = None  # (x, y) normalized 0-1
        self.coin_apparent_pos = None
        self.coin_radius = 0.03  # Normalized radius
        self.coin_detected = False
        
        # Detection state
        self.detection_mode = False
        
        # GPU
        self.gpu = GPUAccelerator()
        
        self.setup_ui()
        self.warmup_jit()
        self.render_thread = threading.Thread(target=self.render_loop, daemon=True)
        self.render_thread.start()

    def setup_ui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.root = ctk.CTk()
        self.root.title("Caustics Pro v2.0 - Advanced Water Optics")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)
        
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Preview
        self.preview_frame = ctk.CTkFrame(self.main_frame)
        self.preview_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        self.preview_header = ctk.CTkLabel(self.preview_frame, text=" Real-Time Preview ", font=ctk.CTkFont(size=16, weight="bold"))
        self.preview_header.pack(pady=(10, 5))
        
        self.canvas_frame = ctk.CTkFrame(self.preview_frame, fg_color="black")
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.preview_label = ctk.CTkLabel(self.canvas_frame, text="")
        self.preview_label.pack(fill="both", expand=True)
        self.preview_label.bind("<Button-1>", self.on_preview_click)
        
        # Info bar
        info_frame = ctk.CTkFrame(self.preview_frame, fg_color="transparent")
        info_frame.pack(fill="x", padx=10)
        self.fps_label = ctk.CTkLabel(info_frame, text="FPS: --", font=ctk.CTkFont(size=12))
        self.fps_label.pack(side="left")
        self.gpu_label = ctk.CTkLabel(info_frame, text=f"GPU: {self.gpu.get_device_name()}", font=ctk.CTkFont(size=12), text_color="cyan")
        self.gpu_label.pack(side="right")
        
        # Right panel - Controls
        self.control_frame = ctk.CTkFrame(self.main_frame, width=400)
        self.control_frame.pack(side="right", fill="y")
        self.control_frame.pack_propagate(False)
        
        self.title_label = ctk.CTkLabel(self.control_frame, text=" CAUSTICS PRO v2", font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.pack(pady=(10, 5))
        
        # Preset dropdown
        preset_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        preset_frame.pack(fill="x", padx=15, pady=5)
        ctk.CTkLabel(preset_frame, text="Preset:").pack(side="left")
        self.preset_var = ctk.StringVar(value="Calm Pool")
        self.preset_dropdown = ctk.CTkOptionMenu(preset_frame, variable=self.preset_var, values=list(PRESETS.keys()), command=self.on_preset_change)
        self.preset_dropdown.pack(side="right", fill="x", expand=True, padx=(10, 0))
        
        self.sliders = {}
        
        # Tabview
        self.tabview = ctk.CTkTabview(self.control_frame, height=450)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=5)
        
        self.tab_waves = self.tabview.add("Waves")
        self.tab_lights = self.tabview.add("Lights")
        self.tab_render = self.tabview.add("Render")
        self.tab_post = self.tabview.add("Post-FX")
        self.tab_detect = self.tabview.add("Detect")
        
        # WAVES TAB
        self.create_slider(self.tab_waves, "Wave Scale", "wave_scale", 0.5, 10.0)
        self.create_slider(self.tab_waves, "Wave Height", "wave_height", 0.01, 1.0)
        self.create_slider(self.tab_waves, "Micro Scale", "micro_scale", 10.0, 150.0)
        self.create_slider(self.tab_waves, "Micro Height", "micro_height", 0.0, 0.2)
        self.create_slider(self.tab_waves, "Gerstner Amp", "gerstner_amp", 0.0, 0.5)
        self.create_slider(self.tab_waves, "Gerstner λ", "gerstner_wl", 0.5, 10.0)
        self.create_slider(self.tab_waves, "Gerstner Speed", "gerstner_speed", 0.1, 3.0)
        
        # LIGHTS TAB
        ctk.CTkLabel(self.tab_lights, text="Light 1 (Primary)", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.create_slider(self.tab_lights, "Angle X", "light1_x", -45.0, 45.0)
        self.create_slider(self.tab_lights, "Angle Y", "light1_y", -45.0, 45.0)
        self.create_slider(self.tab_lights, "Intensity", "light1_intensity", 0.0, 2.0)
        ctk.CTkLabel(self.tab_lights, text="Light 2 (Secondary)", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.create_slider(self.tab_lights, "Angle X", "light2_x", -45.0, 45.0)
        self.create_slider(self.tab_lights, "Angle Y", "light2_y", -45.0, 45.0)
        self.create_slider(self.tab_lights, "Intensity", "light2_intensity", 0.0, 2.0)
        
        # RENDER TAB
        self.create_slider(self.tab_render, "Water Depth", "water_depth", 0.5, 10.0)
        self.create_slider(self.tab_render, "Exposure", "exposure", 0.5, 4.0)
        self.pause_btn = ctk.CTkButton(self.tab_render, text=" Pause", command=self.toggle_pause)
        self.pause_btn.pack(fill="x", padx=10, pady=10)
        
        # POST-FX TAB
        self.create_slider(self.tab_post, "Bloom Threshold", "bloom_threshold", 0.0, 1.0)
        self.create_slider(self.tab_post, "Bloom Intensity", "bloom_intensity", 0.0, 1.5)
        self.create_slider(self.tab_post, "Vignette", "vignette", 0.0, 0.8)
        self.create_slider(self.tab_post, "SSS Intensity", "sss_intensity", 0.0, 1.0)
        self.create_slider(self.tab_post, "SSS Radius", "sss_radius", 3, 25)
        
        # DETECT TAB - Coin simulation & detection
        coin_frame = ctk.CTkFrame(self.tab_detect, fg_color="#1a2a1a")
        coin_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(coin_frame, text="🪙 Coin Simulation", font=ctk.CTkFont(weight="bold")).pack(pady=3)
        coin_btns = ctk.CTkFrame(coin_frame, fg_color="transparent")
        coin_btns.pack(fill="x", padx=5, pady=3)
        self.drop_coin_btn = ctk.CTkButton(coin_btns, text="Drop Coin", command=self.drop_coin, width=90, fg_color="#3d5a27")
        self.drop_coin_btn.pack(side="left", padx=2)
        self.random_coin_btn = ctk.CTkButton(coin_btns, text="Random", command=self.random_coin, width=70, fg_color="#27405a")
        self.random_coin_btn.pack(side="left", padx=2)
        self.clear_coin_btn = ctk.CTkButton(coin_btns, text="Clear", command=self.clear_coin, width=60, fg_color="#5a2727")
        self.clear_coin_btn.pack(side="left", padx=2)
        self.coin_status = ctk.CTkLabel(coin_frame, text="No coin in pool", text_color="gray", font=ctk.CTkFont(size=11))
        self.coin_status.pack(pady=2)
        
        # Detection controls
        detect_frame = ctk.CTkFrame(self.tab_detect, fg_color="#1a1a2a")
        detect_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(detect_frame, text="🔍 Object Detection", font=ctk.CTkFont(weight="bold")).pack(pady=3)
        detect_btns = ctk.CTkFrame(detect_frame, fg_color="transparent")
        detect_btns.pack(fill="x", padx=5, pady=3)
        self.detect_btn = ctk.CTkButton(detect_btns, text="Enable", command=self.toggle_detection, width=80, fg_color="#2d4a5a")
        self.detect_btn.pack(side="left", padx=2)
        self.reload_btn = ctk.CTkButton(detect_btns, text="🔄 Reload", command=self.reload_detection, width=80, fg_color="#4a4a2d")
        self.reload_btn.pack(side="left", padx=2)
        self.detect_coin_btn = ctk.CTkButton(detect_btns, text="Find Coin", command=self.detect_coin_position, width=80, fg_color="#2d5a4a")
        self.detect_coin_btn.pack(side="left", padx=2)
        
        self.detect_result = ctk.CTkTextbox(self.tab_detect, height=130, font=ctk.CTkFont(family="Consolas", size=10))
        self.detect_result.pack(fill="x", padx=5, pady=3)
        self.detect_result.insert("1.0", "Drop a coin, then use detection\nto find its real vs apparent position.")
        self.detect_result.configure(state="disabled")
        
        # Compact action buttons
        action_frame = ctk.CTkFrame(self.control_frame, fg_color="transparent")
        action_frame.pack(fill="x", padx=10, pady=3)
        btn_row1 = ctk.CTkFrame(action_frame, fg_color="transparent")
        btn_row1.pack(fill="x")
        self.render_4k_btn = ctk.CTkButton(btn_row1, text="🎨 4K", command=self.render_4k, width=80, fg_color="#2d5a27")
        self.render_4k_btn.pack(side="left", padx=2, pady=2)
        self.export_btn = ctk.CTkButton(btn_row1, text="🎬 Video", command=self.export_video, width=80, fg_color="#5a2727")
        self.export_btn.pack(side="left", padx=2, pady=2)
        self.clear_ripples_btn = ctk.CTkButton(btn_row1, text="🌊 Clear", command=self.clear_ripples, width=80)
        self.clear_ripples_btn.pack(side="left", padx=2, pady=2)
        self.pause_compact_btn = ctk.CTkButton(btn_row1, text="⏸", command=self.toggle_pause, width=40)
        self.pause_compact_btn.pack(side="left", padx=2, pady=2)
        
        self.status_label = ctk.CTkLabel(self.control_frame, text="Ready", font=ctk.CTkFont(size=10), text_color="gray")
        self.status_label.pack(pady=(2, 5))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def create_slider(self, parent, label, key, min_v, max_v):
        frame = ctk.CTkFrame(parent, fg_color="transparent", height=28)
        frame.pack(fill="x", padx=3, pady=1)
        frame.pack_propagate(False)
        unit = UNITS.get(key, "")
        lbl_text = f"{label} [{unit}]" if unit else label
        lbl = ctk.CTkLabel(frame, text=lbl_text, width=95, anchor="w", font=ctk.CTkFont(size=11))
        lbl.pack(side="left")
        val = self.params.get(key, min_v)
        val_lbl = ctk.CTkLabel(frame, text=f"{val:.2f}", width=40, font=ctk.CTkFont(size=10))
        val_lbl.pack(side="right")
        slider = ctk.CTkSlider(frame, from_=min_v, to=max_v, number_of_steps=100, height=14, command=lambda v, k=key, vl=val_lbl: self.on_slider_change(k, v, vl))
        slider.set(val)
        slider.pack(side="right", fill="x", expand=True, padx=2)
        self.sliders[key] = (slider, val_lbl)

    def on_slider_change(self, key, value, vlbl):
        self.params[key] = float(value)
        vlbl.configure(text=f"{value:.2f}")

    def on_preset_change(self, name):
        if name in PRESETS:
            self.params = dict(PRESETS[name])
            for k, (s, v) in self.sliders.items():
                if k in self.params:
                    s.set(self.params[k])
                    v.configure(text=f"{self.params[k]:.2f}")

    def toggle_pause(self):
        self.paused = not self.paused
        try:
            self.pause_btn.configure(text="▶ Resume" if self.paused else "⏸ Pause")
        except: pass
        try:
            self.pause_compact_btn.configure(text="▶" if self.paused else "⏸")
        except: pass

    def toggle_detection(self):
        self.detection_mode = not self.detection_mode
        self.detect_btn.configure(text="ON" if self.detection_mode else "Enable", fg_color="#5a2d2d" if self.detection_mode else "#2d4a5a")
        if self.detection_mode:
            self.status_label.configure(text="Click pool to detect position")

    def clear_ripples(self):
        self.ripple_times[:] = -100
        self.ripple_count = 0

    # ─── COIN SIMULATION ───
    def drop_coin(self):
        self.coin_real_pos = (0.5, 0.5)
        self._update_coin_apparent()
    
    def random_coin(self):
        self.coin_real_pos = (np.random.uniform(0.15, 0.85), np.random.uniform(0.15, 0.85))
        self._update_coin_apparent()
    
    def clear_coin(self):
        self.coin_real_pos = None
        self.coin_apparent_pos = None
        self.coin_detected = False
        self.coin_status.configure(text="No coin in pool", text_color="gray")
    
    def _update_coin_apparent(self):
        if self.coin_real_pos is None:
            return
        depth = self.params["water_depth"]
        ior = float(cauchy_ior(WAVELENGTH_G))
        rx, ry = self.coin_real_pos
        # Apparent position is shifted due to refraction
        shift_factor = 1.0 - (1.0 / ior)
        ax = rx + (rx - 0.5) * shift_factor * (depth / 5.0)
        ay = ry + (ry - 0.5) * shift_factor * (depth / 5.0)
        self.coin_apparent_pos = (ax, ay)
        self.coin_detected = False
        self.coin_status.configure(text=f"Coin at ({rx*100:.0f}, {ry*100:.0f}) cm - Find it!", text_color="#80ff80")
    
    def detect_coin_position(self):
        if self.coin_real_pos is None:
            self._show_detection_result("No coin in pool!\nDrop a coin first.")
            return
        self._update_coin_apparent()
        rx, ry = self.coin_real_pos
        ax, ay = self.coin_apparent_pos
        depth = self.params["water_depth"]
        ior = float(cauchy_ior(WAVELENGTH_G))
        disp = np.sqrt((rx-ax)**2 + (ry-ay)**2) * 100
        result = f"=== COIN DETECTED ===\n\nReal Position:\n  X: {rx*100:.1f} cm  Y: {ry*100:.1f} cm\n\nApparent Position:\n  X: {ax*100:.1f} cm  Y: {ay*100:.1f} cm\n\nRefraction shift: {disp:.2f} cm\nDepth: {depth:.2f} m | IOR: {ior:.4f}"
        self._show_detection_result(result)
        self.coin_detected = True
        self.coin_status.configure(text="Coin detected ✓", text_color="cyan")
    
    def reload_detection(self):
        """Reload/reset detection for a new measurement."""
        self.coin_detected = False
        if self.coin_real_pos:
            self._update_coin_apparent()
            self._show_detection_result("Detection reset.\nCoin repositioned - detect again!")
        else:
            self._show_detection_result("Ready for new detection.\nDrop a coin or click on pool.")
        self.status_label.configure(text="Detection reloaded")
    
    def _show_detection_result(self, text):
        self.detect_result.configure(state="normal")
        self.detect_result.delete("1.0", "end")
        self.detect_result.insert("1.0", text)
        self.detect_result.configure(state="disabled")

    def on_preview_click(self, event):
        x = event.x / self.preview_width
        y = event.y / self.preview_height
        if self.detection_mode:
            self.perform_detection(x, y)
        else:
            idx = self.ripple_count % self.max_ripples
            self.ripple_centers[idx, 0] = x
            self.ripple_centers[idx, 1] = y
            self.ripple_times[idx] = self.time
            self.ripple_count += 1

    def perform_detection(self, apparent_x, apparent_y):
        depth = self.params["water_depth"]
        ior = float(cauchy_ior(WAVELENGTH_G))
        real_x, real_y, real_z, dx, dy = find_real_object_position(apparent_x - 0.5, apparent_y - 0.5, depth, 0.0, 2.0, ior)
        displacement = np.sqrt(dx**2 + dy**2) * 100
        self.detect_result.configure(state="normal")
        self.detect_result.delete("1.0", "end")
        result = f"=== DETECTION RESULT ===\n\nApparent Position:\n  X: {apparent_x*100:.1f} cm\n  Y: {apparent_y*100:.1f} cm\n\nReal Position (Snell corrected):\n  X: {(real_x+0.5)*100:.1f} cm\n  Y: {(real_y+0.5)*100:.1f} cm\n  Z: {real_z*100:.1f} cm (depth)\n\nDisplacement: {displacement:.2f} cm\nIOR used: {ior:.4f}\nWater depth: {depth:.2f} m"
        self.detect_result.insert("1.0", result)
        self.detect_result.configure(state="disabled")

    def warmup_jit(self):
        self.status_label.configure(text="JIT compiling...")
        self.root.update()
        test = np.zeros((32, 32), dtype=np.float32)
        rc = np.zeros((2, 2), dtype=np.float32)
        rt = np.zeros(2, dtype=np.float32) - 100
        generate_water_surface_with_ripples(32, 32, 0.0, self.perm, 3.0, 0.1, 40.0, 0.02, 0.05, 3.0, 0.5, rc, rt, 0)
        lights = np.array([[0.1, 0.1, -0.99]], dtype=np.float32)
        colors = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        ray_trace_kernel_multilight(test, test.copy(), test.copy(), test.copy(), 32, 32, 2.0, 0.05, lights, colors, 1)
        self.status_label.configure(text="Ready - JIT compiled ")

    def get_lights(self):
        lights = []
        colors = []
        for i in [1, 2]:
            intensity = self.params.get(f"light{i}_intensity", 0)
            if intensity > 0.01:
                ax = np.radians(self.params.get(f"light{i}_x", 0))
                ay = np.radians(self.params.get(f"light{i}_y", 0))
                dx = np.sin(ax) * np.cos(ay)
                dy = np.sin(ay)
                dz = -np.cos(ax) * np.cos(ay)
                l = np.sqrt(dx**2 + dy**2 + dz**2)
                lights.append([dx/l, dy/l, dz/l])
                colors.append([intensity, intensity, intensity])
        if not lights:
            lights.append([0.1, 0.1, -0.99])
            colors.append([1.0, 1.0, 1.0])
        return np.array(lights, dtype=np.float32), np.array(colors, dtype=np.float32)

    def render_frame(self, width=None, height=None, supersample=1):
        w = width or self.sim_width
        h = height or self.sim_height
        rw, rh = w * supersample, h * supersample
        surface = generate_water_surface_with_ripples(rw, rh, self.time, self.perm, np.float32(self.params["wave_scale"]), np.float32(self.params["wave_height"]), np.float32(self.params["micro_scale"]), np.float32(self.params["micro_height"]), np.float32(self.params["gerstner_amp"]), np.float32(self.params["gerstner_wl"]), np.float32(self.params["gerstner_speed"]), self.ripple_centers, self.ripple_times, min(self.ripple_count, self.max_ripples))
        cr = np.zeros((rh, rw), dtype=np.float32)
        cg = np.zeros((rh, rw), dtype=np.float32)
        cb = np.zeros((rh, rw), dtype=np.float32)
        lights, colors = self.get_lights()
        ray_trace_kernel_multilight(surface, cr, cg, cb, rw, rh, np.float32(self.params["water_depth"]), self.cell_size, lights, colors, len(lights))
        if supersample > 1:
            cr = cv2.resize(cr, (w, h), interpolation=cv2.INTER_AREA)
            cg = cv2.resize(cg, (w, h), interpolation=cv2.INTER_AREA)
            cb = cv2.resize(cb, (w, h), interpolation=cv2.INTER_AREA)
        return post_process(cr, cg, cb, self.params["bloom_threshold"], self.params["bloom_intensity"], self.params["vignette"], self.params["exposure"], self.params.get("sss_intensity", 0), self.params.get("sss_radius", 7))

    def render_loop(self):
        ftimes = []
        while self.running:
            if self.paused or self.rendering_4k or self.exporting_video:
                time.sleep(0.05)
                continue
            st = time.perf_counter()
            img = self.render_frame()
            preview = cv2.resize(img, (self.preview_width, self.preview_height), interpolation=cv2.INTER_LINEAR)
            
            # Draw coin if present
            if self.coin_real_pos is not None:
                rx, ry = self.coin_real_pos
                coin_r = int(self.coin_radius * self.preview_width)
                # Real position (green circle)
                cx_real = int(rx * self.preview_width)
                cy_real = int(ry * self.preview_height)
                cv2.circle(preview, (cx_real, cy_real), coin_r, (0, 200, 0), 2)
                cv2.circle(preview, (cx_real, cy_real), coin_r//3, (0, 255, 0), -1)
                # Apparent position (red dashed-like outline)
                if self.coin_apparent_pos:
                    ax, ay = self.coin_apparent_pos
                    cx_app = int(ax * self.preview_width)
                    cy_app = int(ay * self.preview_height)
                    cv2.circle(preview, (cx_app, cy_app), coin_r + 3, (0, 0, 200), 1)
                    # Line connecting real to apparent
                    cv2.line(preview, (cx_real, cy_real), (cx_app, cy_app), (100, 100, 255), 1)
            
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            self.root.after(0, self.update_preview, preview_rgb)
            self.time += 0.033
            ft = time.perf_counter() - st
            ftimes.append(ft)
            if len(ftimes) > 30: ftimes.pop(0)
            fps = 1.0 / (sum(ftimes) / len(ftimes))
            self.root.after(0, self.update_fps, fps)
            elapsed = time.perf_counter() - st
            if elapsed < 0.016: time.sleep(0.016 - elapsed)

    def update_preview(self, img):
        try:
            pil = Image.fromarray(img)
            photo = ImageTk.PhotoImage(pil)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
        except: pass

    def update_fps(self, fps):
        self.fps_label.configure(text=f"FPS: {fps:.1f}")

    def render_4k(self):
        if self.rendering_4k: return
        self.rendering_4k = True
        self.status_label.configure(text="Rendering 4K...")
        self.render_4k_btn.configure(state="disabled")
        def do():
            try:
                img = self.render_frame(1024, 1024, 4)
                img = cv2.resize(img, (4096, 4096), interpolation=cv2.INTER_LANCZOS4)
                fn = f"caustics_4k_{int(time.time())}.png"
                cv2.imwrite(fn, img)
                self.root.after(0, lambda: self.status_label.configure(text=f"Saved: {fn}"))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.configure(text=f"Error: {e}"))
            finally:
                self.rendering_4k = False
                self.root.after(0, lambda: self.render_4k_btn.configure(state="normal"))
        threading.Thread(target=do, daemon=True).start()

    def export_video(self):
        if self.exporting_video: return
        self.exporting_video = True
        self.status_label.configure(text="Exporting video...")
        self.export_btn.configure(state="disabled")
        def do():
            try:
                fn = f"caustics_{int(time.time())}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                w = cv2.VideoWriter(fn, fourcc, 30, (512, 512))
                orig_t = self.time
                for i in range(150):
                    self.time = (i / 150) * 2 * np.pi
                    img = self.render_frame(512, 512)
                    w.write(img)
                    self.root.after(0, lambda p=int((i+1)/150*100): self.status_label.configure(text=f"Export: {p}%"))
                w.release()
                self.time = orig_t
                self.root.after(0, lambda: self.status_label.configure(text=f"Saved: {fn}"))
            except Exception as e:
                self.root.after(0, lambda: self.status_label.configure(text=f"Error: {e}"))
            finally:
                self.exporting_video = False
                self.root.after(0, lambda: self.export_btn.configure(state="normal"))
        threading.Thread(target=do, daemon=True).start()

    def on_close(self):
        self.running = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()

def main():
    app = CausticsProApp()
    app.run()

if __name__ == "__main__":
    main()


