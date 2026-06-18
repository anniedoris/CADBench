import os
import tempfile
from fpdf import FPDF
import visualize_mesh_davinci

# --- Configuration ---
BASE_DIR = "/home/jacob/CADBench/data"

# Deep Matte Palette (Saturated, Low Brightness)
THEMES = {
    "bench0": {
        "easy":   ("#a52a2a", "#d9a5a5"), # Deep Matte Red
        "medium": ("#a52a2a", "#c06666"), 
        "hard":   ("#7b241c", "#e6b0aa"), 
    },
    "bench0F": {
        "easy":   ("#2e5984", "#a3c1da"), # Deep Matte Blue
        "medium": ("#2e5984", "#7ea4c4"), 
        "hard":   ("#1b4f72", "#a9cce3"), 
    },
    "bench1A": {
        "easy":   ("#2d5a27", "#aed581"), # Deep Matte Green
        "medium": ("#2d5a27", "#8bc34a"),
        "hard":   ("#1b3917", "#c5e1a5"), 
    },
    "bench1B": {
        "easy":   ("#a67c00", "#fff176"), # Deep Matte Gold
        "medium": ("#a67c00", "#ffd54f"),
        "hard":   ("#7d5d00", "#ffe082"), 
    },
    "bench2": {
        "unlabeled": ("#804a7e", "#d2b4de"), # Deep Matte Magenta
    },
    "bench3": {
        "unlabeled": ("#bf6000", "#ffcc80"), # Deep Matte Orange
    }
}

def get_theme_colors(bench, category):
    try:
        return THEMES[bench][category]
    except KeyError:
        return ("#a52a2a", "#d9a5a5")

def generate_extended_strip(bench, category, file_id, output_folder="reports"):
    sub_path = f"{bench}/{category}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_name = os.path.join(output_folder, f"full_strip_{bench}_{category}_{file_id}.pdf")
    full_root = os.path.join(BASE_DIR, sub_path)
    clean_color, noisy_color = get_theme_colors(bench, category)

    # Standardizing the 5 paths
    paths = {
        "stl":       os.path.join(full_root, "stl", f"{file_id}.stl"),
        "noisy":     os.path.join(full_root, "noisy_stl", f"{file_id}_noisy.stl"),
        "gray":      os.path.join(full_root, "singleview_image", f"{file_id}_0.png"),
        "pbr":       os.path.join(full_root, "pbr", f"{file_id}.png"),
        "multiview": os.path.join(full_root, "multiview_image", f"{file_id}.png") # No _0 usually
    }

    tmp1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    render_out, noisy_out = tmp1.name, tmp2.name
    tmp1.close(); tmp2.close()

    try:
        # Rendering
        if os.path.exists(paths["stl"]):
            visualize_mesh_davinci.visualize_mesh_cad(paths["stl"], style = "mesh", color=clean_color, output_image=render_out)
        if os.path.exists(paths["noisy"]):
            visualize_mesh_davinci.visualize_mesh_cad(paths["noisy"], style = "mesh", color=noisy_color, output_image=noisy_out)

        # --- UPDATED PDF DIMENSIONS ---
        # Height remains 70mm.
        # Width increases: 5 images * 70mm = 350mm total.
        pdf = FPDF(orientation='L', unit='mm', format=(70, 350))
        pdf.set_margins(0, 0, 0)
        pdf.add_page()

        # Image sequence
        items = [render_out, noisy_out, paths["gray"], paths["pbr"], paths["multiview"]]
        
        x_offset = 0
        for img_path in items:
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                pdf.image(img_path, x=x_offset, y=0, h=70)
            x_offset += 70

        pdf.output(output_name)
        print(f"✅ Generated 5-Image Strip: {output_name}")

    finally:
        for f in [render_out, noisy_out]:
            if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    # Test batch including multiview
    jobs = [
        ("bench2", "unlabeled", "00026168"),

    ]
    
    for b, c, f in jobs:
        generate_extended_strip(b, c, f)