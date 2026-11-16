import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont

assets_dir = Path("submission_package/assets")
assets_dir.mkdir(parents=True, exist_ok=True)

print("=== 1. GENERATING CONFUSION MATRIX ASSET ===")
cm = np.array([
    [7, 0, 0, 1],
    [4, 0, 2, 2],
    [6, 0, 2, 0],
    [6, 0, 0, 2]
])
words = ["bin", "lay", "place", "set"]

fig, ax = plt.subplots(figsize=(5, 4.2), dpi=200)
cax = ax.matshow(cm, cmap="Blues", alpha=0.85)

for i in range(len(words)):
    for j in range(len(words)):
        color = "white" if cm[i, j] > 3 else "black"
        ax.text(j, i, str(cm[i, j]), va="center", ha="center", fontsize=11, fontweight="bold", color=color)

fig.colorbar(cax, fraction=0.046, pad=0.04)
ax.set_xticks(range(len(words)))
ax.set_yticks(range(len(words)))
ax.set_xticklabels(words, fontsize=10, fontweight="bold")
ax.set_yticklabels(words, fontsize=10, fontweight="bold")
ax.set_xlabel("Predicted Class", labelpad=10, fontsize=10, fontweight="bold")
ax.set_ylabel("True Class", labelpad=10, fontsize=10, fontweight="bold")
ax.set_title("Held-Out Test Confusion Matrix (s30-s34)\nTop-1: 34.38% | Top-2: 62.50%", pad=15, fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(assets_dir / "confusion_matrix.png")
plt.close()
print("  Saved confusion_matrix.png")

print("\n=== 2. GENERATING SYSTEM ARCHITECTURE DIAGRAM ===")
fig, ax = plt.subplots(figsize=(8, 4), dpi=200)
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)

# Boxes
def draw_box(x, y, w, h, text, subtext, color="#1E293B", edge="#38BDF8"):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.15",
                                 facecolor=color, edgecolor=edge, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + 0.15, text, ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(x + w/2, y + h/2 - 0.2, subtext, ha="center", va="center", color="#94A3B8", fontsize=7.5)

def draw_arrow(x1, y1, x2, y2, label=""):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=1.5, mutation_scale=12))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.15, label, ha="center", va="bottom", color="#CBD5E1", fontsize=7)

draw_box(0.5, 3.2, 2.2, 1.2, "Video Acquisition", "Webcam / Silent MP4\n(24 fps, BGR)", "#0F172A", "#38BDF8")
draw_box(3.8, 3.2, 2.4, 1.2, "MediaPipe Tracking", "468 Facial Mesh Landmarks\nROI Bounding Box", "#0F172A", "#38BDF8")
draw_box(7.2, 3.2, 2.3, 1.2, "Mouth Normalization", "64x64 Grayscale Crop\nTemporal Resampling T=24", "#0F172A", "#38BDF8")

draw_box(7.2, 0.8, 2.3, 1.4, "WordVideoNet Model", "Conv3D (3 Blocks) +\nBiGRU (2 Layers, 48 hidden)", "#0F172A", "#10B981")
draw_box(3.8, 0.8, 2.4, 1.4, "Uncertainty Gating", "Softmax Probabilities\nThreshold tau = 0.35", "#0F172A", "#10B981")
draw_box(0.5, 0.8, 2.2, 1.4, "React/Vite Frontend", "Real-Time Word Pill\nAudio Feedback / TTS", "#0F172A", "#10B981")

draw_arrow(2.7, 3.8, 3.8, 3.8, "Raw Video")
draw_arrow(6.2, 3.8, 7.2, 3.8, "Lip Landmarks")
draw_arrow(8.35, 3.2, 8.35, 2.2, "Mouth Tensor [1,24,64,64]")
draw_arrow(7.2, 1.5, 6.2, 1.5, "Logits [1,4]")
draw_arrow(3.8, 1.5, 2.7, 1.5, "Gated Word")

plt.tight_layout()
plt.savefig(assets_dir / "system_architecture.png")
plt.close()
print("  Saved system_architecture.png")

print("\n=== 3. GENERATING LOGO BADGES ===")
# KJSSE Badge
img_kjsse = Image.new("RGBA", (400, 100), (255, 255, 255, 0))
d1 = ImageDraw.Draw(img_kjsse)
d1.rounded_rectangle([5, 5, 395, 95], radius=10, fill=(240, 244, 248), outline=(180, 20, 20), width=2)
d1.rectangle([15, 15, 35, 85], fill=(180, 20, 20))
d1.text((45, 20), "K. J. SOMAIYA", fill=(180, 20, 20), font_size=18)
d1.text((45, 42), "School of Engineering", fill=(30, 41, 59), font_size=13)
d1.text((45, 62), "Somaiya Vidyavihar University", fill=(100, 116, 139), font_size=11)
img_kjsse.save(assets_dir / "college_logo.png")
print("  Saved college_logo.png")

# Somaiya Trust Blue Logo Badge
img_trust = Image.new("RGBA", (400, 100), (255, 255, 255, 0))
d2 = ImageDraw.Draw(img_trust)
d2.rounded_rectangle([5, 5, 395, 95], radius=10, fill=(240, 244, 255), outline=(26, 54, 93), width=2)
d2.rectangle([15, 15, 35, 85], fill=(26, 54, 93))
d2.text((45, 20), "SOMAIYA TRUST", fill=(26, 54, 93), font_size=18)
d2.text((45, 42), "Knowledge Alone Liberates", fill=(30, 41, 59), font_size=13)
d2.text((45, 62), "Trust & Vidyavihar Network", fill=(100, 116, 139), font_size=11)
img_trust.save(assets_dir / "somaiya_trust_logo.png")
print("  Saved somaiya_trust_logo.png")

print("\nAsset generation completed successfully!")
