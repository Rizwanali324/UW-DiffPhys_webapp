import os
import subprocess
from flask import Flask, request, jsonify, send_file
from pyngrok import ngrok

# Set the auth token for ngrok
ngrok.set_auth_token("2oclGQq2wjj2QBvfFqKaxdk7iuQ_bD9mhRPSqovvrGZb5ZP9")

# Initialize Flask app
app = Flask(__name__)

# Ngrok tunnel for exposing the Flask app
public_url = ngrok.connect(5000)
print(f"Flask app is live at {public_url}")

# Directory where results will be saved
output_dir = '/results'

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/run_inference', methods=['POST'])
def run_inference():
    # Check if 'condition_image' is in the request data
    if 'condition_image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    # Check if more than one file is uploaded
    if len(request.files.getlist('condition_image')) > 1:
        return jsonify({'error': 'Only one image can be uploaded at a time'}), 400
    
    # Get the image file from the request
    image_file = request.files['condition_image']
    
    # Validate the image format
    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Invalid image format. Allowed formats are PNG, JPG, JPEG.'}), 400
    
    # Save the uploaded image
    image_path = f"/content/UW-DiffPhys_webapp/{image_file.filename}"
    image_file.save(image_path)
    
    # Construct the command with the given arguments
    command = [
        "python", "/inference_UW-DDIM.py",
        "--config", "/configs/underwater_lsui_uieb_128.yml",
        "--resume", "/content/Checkpoints",
        "--sampling_timesteps", "25",
        "--eta", "0",
        "--condition_image", image_path,
        "--seed", "5"
    ]
    
    # Execute the command
    try:
        subprocess.run(command, check=True)
        
        # Get the result file (assuming it follows a naming convention based on the input image)
        result_file = os.path.join(output_dir, f"{os.path.splitext(image_file.filename)[0]}.png")
        
        if os.path.exists(result_file):
            # Return the result image directly
            response = send_file(result_file, mimetype='image/png')
            
            # Delete the input image after serving the result
            if os.path.exists(image_path):
                os.remove(image_path)
            
            return response
        else:
            return jsonify({'error': 'Result file not found'}), 500
    
    except subprocess.CalledProcessError as e:
        return jsonify({'error': f'Error running inference: {str(e)}'}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(port=5000)
