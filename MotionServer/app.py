import json
from flask import Flask, request, jsonify
from utils import save_json, load_json, get_transparent_img_bytes, timer
from points2image import draw_base_flower
 
app = Flask(__name__)
request_count = 0

unity_points = []
current_points = []
pen_up_counter = 0
image_bytes = get_transparent_img_bytes()
image = None






@app.route('/receive', methods=['PUT'])
def receive_data():
    global request_count, unity_points
    global current_points, pen_up_counter, image_bytes, image

    request_count += 1
    # Get the JSON data from the request
    data = request.get_json()
    unity_points.append(data)
    # print(request_count, data)
 
    # Extract start and end positions from the JSON data
    start_x = data['startX']
    start_y = data['startY']
    end_x = data['endX']
    end_y = data['endY']
    pen_touch = data['penTouchingPaper']
 
    if pen_touch:
        # Reset counter and add the point to the current line
        pen_up_counter = 0
        if len(current_points) == 0 or (current_points[-1][0] != data['startX'] and current_points[-1][1] != data['startY']):
            current_points.append((data['startX'], data['startY']))  # Assuming points are tuples (x, y)
            current_points.append((data['endX'], data['endY']))
    else:
        pen_up_counter += 1
        # Check if the pen was up for 5 consecutive times
        if pen_up_counter >= 5 and current_points:
            # Add the current line to unity_points and reset for next line
            pen_up_counter = 0

            # Generate image based on accumulated points and convert to bytes
            with timer("Draw flower"):
                image_bytes, center_x, center_y = draw_base_flower(current_points)
            # image, center_x, center_y = draw_base_flower(current_points, resize=False, byte_format=False)
            # image.show()
            # Reset unity_points for next line collection
            current_points = []
            # Respond with image bytes (Here, just indicating success for simplicity)
            return jsonify({'message': 'Image generated and sent successfully'})
    
    # Default response for when no image is generated
    return jsonify({'message': 'Data received successfully'})
 


@app.route('/get_image', methods=['GET'])
def get_flower_image():
    global image_bytes
    # if not image_bytes:
    #     image_bytes = "Default"
    # print(image_bytes)
    return image_bytes



if __name__ == '__main__':
    app.run(host='localhost', port=5000) # Change host and port as needed
    # save_json(unity_points, "../Data_Unity/3linesR2WSID720.txt")
    