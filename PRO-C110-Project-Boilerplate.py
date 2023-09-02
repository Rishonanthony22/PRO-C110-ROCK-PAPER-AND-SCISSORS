import cv2

# Initialize the video capture (0 is typically the default camera, change if needed)
cap = cv2.VideoCapture(0)

# Load the pre-trained Haar Cascade Classifiers for Rock, Paper, and Scissors gestures
rock_cascade = cv2.CascadeClassifier('rock.xml')
paper_cascade = cv2.CascadeClassifier('paper.xml')
scissors_cascade = cv2.CascadeClassifier('scissors.xml')

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Resize and normalize the frame
    frame = cv2.resize(frame, (224, 224))  # Adjust the dimensions as needed
    frame = frame / 255.0  # Normalize pixel values to the range [0, 1]

    # Convert the frame to grayscale for gesture detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Rock gestures
    rock_gestures = rock_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in rock_gestures:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Rock', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Detect Paper gestures
    paper_gestures = paper_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in paper_gestures:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'Paper', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Detect Scissors gestures
    scissors_gestures = scissors_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in scissors_gestures:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Scissors', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
