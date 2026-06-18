import os
import tempfile
import shutil
import gradio as gr

# Import the correct script
import visualize_mesh_davinci

# Set this to your main project folder so you can navigate everywhere inside it
WORKING_DIR = "/home/jacob/CADBench" 

def process_mesh(file_path, mesh_color):
    if not file_path:
        return None, None, "**Currently Rendering:** None", "rendered_mesh.png"
        
    # Extract the filename from the full path to display in the UI
    file_name = os.path.basename(file_path)
    name_without_ext = os.path.splitext(file_name)[0]
    
    # Create the dynamic UI text updates
    display_title = f"**Currently Rendering:** `{file_name}`"
    suggested_save_name = f"{name_without_ext}_render.png"
        
    # Create a temporary file path for the output image
    fd, temp_img_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    
    try:
        # Call your actual script and pass the color from the GUI
        visualize_mesh_davinci.visualize_mesh_cad(
            file_path, 
            style = "mesh",
            color=mesh_color, 
            output_image=temp_img_path   
        )
        
        # Return: 1. Image, 2. Temp path state, 3. Top Label text, 4. Save box text
        return temp_img_path, temp_img_path, display_title, suggested_save_name
    except Exception as e:
        print(f"Error rendering mesh: {e}")
        error_title = f"**Error Rendering:** `{file_name}`\n\n*(Did you remember to update visualize_mesh_davinci.py to accept 'override_color'?)*"
        return None, None, error_title, suggested_save_name

def save_to_server(temp_img_path, save_filename):
    if not temp_img_path or not os.path.exists(temp_img_path):
        return "Error: No image to save. Please render an image first."
    if not save_filename:
        return "Error: Please provide a filename."
        
    try:
        # Save it to the main working directory on the server
        save_path = os.path.join(WORKING_DIR, save_filename)
        shutil.copy(temp_img_path, save_path)
        return f"✅ Successfully saved to server as: \n{save_path}"
    except Exception as e:
        return f"❌ Failed to save: {e}"

# Build the Web GUI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# DaVinci Mesh Visualizer")
    gr.Markdown(f"Browsing remote files starting from: **{WORKING_DIR}**")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Select a Mesh File")
            file_explorer = gr.FileExplorer(
                root_dir=WORKING_DIR,
                file_count="single",
                label="Server Filesystem (Double-click folders to open)"
            )
            
            gr.Markdown("### 2. Render Settings")
            # The Color Picker! Defaults to your original CAD grey.
            color_picker = gr.ColorPicker(label="Mesh Color", value="#bfbeba")
            
            render_btn = gr.Button("Render Image", variant="primary")
        
        with gr.Column():
            # Dynamic label above the image
            current_file_label = gr.Markdown("**Currently Rendering:** None")
            
            output_image = gr.Image(label="Rendered Preview", type="filepath")
            
            # A hidden state to keep track of the temporary image file
            current_temp_path = gr.State()
            
            gr.Markdown("### 3. Save to Server")
            with gr.Row():
                # This textbox will automatically update based on the mesh file
                save_name_input = gr.Textbox(
                    label="Filename", 
                    value="rendered_mesh.png", 
                    scale=2
                )
                save_btn = gr.Button("Save Image", scale=1)
                
            save_status = gr.Textbox(label="Status", interactive=False)

    # Wire up the Render button to pass the color picker value to process_mesh
    render_btn.click(
        fn=process_mesh,
        inputs=[file_explorer, color_picker],
        outputs=[output_image, current_temp_path, current_file_label, save_name_input]
    )
    
    # Wire up the Save button
    save_btn.click(
        fn=save_to_server,
        inputs=[current_temp_path, save_name_input],
        outputs=[save_status]
    )

if __name__ == "__main__":
    # Launch the server
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)