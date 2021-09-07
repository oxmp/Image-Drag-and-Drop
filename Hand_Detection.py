import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Set up the Hands function.
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.8
)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils


hands_video = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5,
)


def detectHandsLandmarks(image, hands, display=True):
    """
    This function performs hands landmarks detection on an image.
    Args:
        image:   The input image with prominent hand(s) whose landmarks needs to be detected.
        hands:   The hands function required to perform the hands landmarks detection.
        display: A boolean value that is if set to true the function displays the original input image, and the output
                 image with hands landmarks drawn and returns nothing.
    Returns:
        output_image: The input image with the detected hands landmarks drawn.
        results: The output of the hands landmarks detection on the input image.
    """

    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # Check if landmarks are found.
    if results.multi_hand_landmarks:

        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(
                image=output_image,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
            )

    # Check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output")
        plt.axis("off")

    # Otherwise
    else:

        # Return the output image and results of hands landmarks detection.
        return output_image, results


def getHandType(image, results, draw=True, display=True):
    """
    This function performs hands type (left or right) classification on hands.
    Args:
        image:   The image of the hands that needs to be classified, with the hands landmarks detection already performed.
        results: The output of the hands landmarks detection performed on the image in which hands types needs
                 to be classified.
        draw:    A boolean value that is if set to true the function writes the hand type label on the output image.
        display: A boolean value that is if set to true the function displays the output image and returns nothing.
    Returns:
        output_image: The image of the hands with the classified hand type label written if it was specified.
        hands_status: A dictionary containing classification info of both hands.
    """

    # Create a copy of the input image to write hand type label on.
    output_image = image.copy()

    # Initialize a dictionary to store the classification info of both hands.
    hands_status = {
        "Right": False,
        "Left": False,
        "Right_index": None,
        "Left_index": None,
    }

    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):

        # Retrieve the label of the found hand.
        hand_type = hand_info.classification[0].label

        # Update the status of the found hand.
        hands_status[hand_type] = True

        # Update the index of the found hand.
        hands_status[hand_type + "_index"] = hand_index

        # Check if the hand type label is specified to be written.
        if draw:

            # Write the hand type on the output image.
            cv2.putText(
                output_image,
                hand_type + " Hand Detected",
                (10, (hand_index + 1) * 30),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 255, 0),
                2,
            )

    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis("off")

    # Otherwise
    else:

        # Return the output image and the hands status dictionary that contains classification info.
        return output_image, hands_status


def drawBoundingBoxes(
    image, results, hand_status, padd_amount=10, draw=True, display=True
):
    """
    This function draws bounding boxes around the hands and write their classified types near them.
    Args:
        image:       The image of the hands on which the bounding boxes around the hands needs to be drawn and the
                     classified hands types labels needs to be written.
        results:     The output of the hands landmarks detection performed on the image on which the bounding boxes needs
                     to be drawn.
        hand_status: The dictionary containing the classification info of both hands.
        padd_amount: The value that specifies the space inside the bounding box between the hand and the box's borders.
        draw:        A boolean value that is if set to true the function draws bounding boxes and write their classified
                     types on the output image.
        display:     A boolean value that is if set to true the function displays the output image and returns nothing.
    Returns:
        output_image:     The image of the hands with the bounding boxes drawn and hands classified types written if it
                          was specified.
        output_landmarks: The dictionary that stores both (left and right) hands landmarks as different elements.
    """

    # Create a copy of the input image to draw bounding boxes on and write hands types labels.
    output_image = image.copy()

    # Initialize a dictionary to store both (left and right) hands landmarks as different elements.
    output_landmarks = {}

    # Get the height and width of the input image.
    height, width, _ = image.shape
    bounding_box = []
    # Iterate over the found hands.
    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
        bbox = []
        # Initialize a list to store the detected landmarks of the hand.
        landmarks = []

        # Iterate over the detected landmarks of the hand.
        for landmark in hand_landmarks.landmark:

            # Append the landmark into the list.
            landmarks.append(
                (
                    int(landmark.x * width),
                    int(landmark.y * height),
                    (landmark.z * width),
                )
            )

        # Get all the x-coordinate values from the found landmarks of the hand.
        x_coordinates = np.array(landmarks)[:, 0]

        # Get all the y-coordinate values from the found landmarks of the hand.
        y_coordinates = np.array(landmarks)[:, 1]

        # Get the bounding box coordinates for the hand with the specified padding.
        x1 = int(np.min(x_coordinates) - padd_amount)
        y1 = int(np.min(y_coordinates) - padd_amount)
        x2 = int(np.max(x_coordinates) + padd_amount)
        y2 = int(np.max(y_coordinates) + padd_amount)
        # print(x1,y1,x2,y2)
        bbox.append([x1, y1, x2, y2])
        # Initialize a variable to store the label of the hand.
        label = "Unknown"

        # Check if the hand we are iterating upon is the right one.
        if hand_status["Right_index"] == hand_index:

            # Update the label and store the landmarks of the hand in the dictionary.
            label = "Right Hand"
            output_landmarks["Right"] = landmarks

        # Check if the hand we are iterating upon is the left one.
        elif hand_status["Left_index"] == hand_index:

            # Update the label and store the landmarks of the hand in the dictionary.
            label = "Left Hand"
            output_landmarks["Left"] = landmarks

        # Check if the bounding box and the classified label is specified to be written.
        if draw:

            # Draw the bounding box around the hand on the output image.
            cv2.rectangle(
                output_image, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8
            )

            # Write the classified label of the hand below the bounding box drawn.
            cv2.putText(
                output_image,
                label,
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (20, 255, 155),
                1,
                cv2.LINE_AA,
            )
        bounding_box.append(bbox)
    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis("off")

    # Otherwise
    else:

        # Return the output image and the landmarks dictionary.
        return output_image, output_landmarks, bounding_box


def hand_data(frame, bdraw=True):
    # Perform Hands landmarks detection.
    frame, results = detectHandsLandmarks(frame, hands_video, display=False)
    # Check if landmarks are found in the frame.

    if results.multi_hand_landmarks:
        lmList_all = []
        # Perform hand(s) type (left or right) classification.
        _, hands_status = getHandType(frame.copy(), results, draw=False, display=False)
        if bdraw == True:
            # Draw bounding boxes around the detected hands and write their classified types near them.
            frame, _, bbox = drawBoundingBoxes(
                frame, results, hands_status, display=False
            )

        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            lmList_all.append(lmList)
    try:
        lmList_all_error = lmList_all
    except:
        lmList_all = ["NULL"]
    try:
        bbox_error = bbox
    except:
        bbox = ["NULL"]
    return frame, lmList_all, bbox


# import cv2
# from Hand_Detection import hand_data
# cap=cv2.VideoCapture(0)
# cap.set(3,1280)
# cap.set(4,720)
# while True:
#     success,img=cap.read()
#     img=cv2.flip(img,1)
#     img,lmList_all,bbox=hand_data(img,bdraw=True)
#     cv2.imshow("VIDEO",img)
#     cv2.waitKey(1)
