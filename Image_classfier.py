
import tensorflow as tf

# Verify GPU availability
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Specify GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# Rest of your TensorFlow code goes here



import cv2
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = InceptionV3(weights='imagenet')

# Define the desired objects to detect
desired_objects = ['human']

# Global variables to track mouse events
mouse_down = False
start_x, start_y = -1, -1
end_x, end_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    global mouse_down, start_x, start_y, end_x, end_y

    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_down = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_down = False
        end_x, end_y = x, y

# Load the video capture
cap = cv2.VideoCapture(0)

# Create a new window
cv2.namedWindow('Live Object Detection')
cv2.setMouseCallback('Live Object Detection', mouse_callback)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    img = cv2.resize(frame, (299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make predictions with the model
    predictions = model.predict(x)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Filter the predictions for desired objects
    filtered_predictions = [pred for pred in decoded_predictions if pred[1] in desired_objects]

    # Display the filtered results
    for pred in filtered_predictions:
        class_label = pred[1]
        score = pred[2]

        # Get the coordinates of the bounding box
        xmin = start_x if start_x < end_x else end_x
        ymin = start_y if start_y < end_y else end_y
        xmax = end_x if start_x < end_x else start_x
        ymax = end_y if start_y < end_y else start_y

        # Draw the bounding box rectangle and label
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, class_label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Draw the crosshair
    cv2.line(frame, (start_x, 0), (start_x, frame.shape[0]), (0, 255, 0), 1)
    cv2.line(frame, (0, start_y), (frame.shape[1], start_y), (0, 255, 0), 1)

    # Show the frame in the window
    cv2.imshow('Live Object Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()
