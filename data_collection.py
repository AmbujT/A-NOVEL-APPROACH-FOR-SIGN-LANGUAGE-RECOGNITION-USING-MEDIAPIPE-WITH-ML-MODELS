import cv2
import os
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3 # denote no of classes to predict = no. of alphabet
dataset_size = 100 # dataset size for each class


for i in range(10):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Camera with index {i} not found.")
        continue

    print(f"Using camera index {i}.")

    for j in range(number_of_classes):
        if not os.path.exists(os.path.join(DATA_DIR, str(j))):
            os.makedirs(os.path.join(DATA_DIR, str(j)))

        print('Collecting data for class {}'.format(j))

        done = False
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                done = True
                break

        if done:
            counter = 0
            while counter < dataset_size:
                ret, frame = cap.read()
                cv2.imshow('frame', frame)
                cv2.waitKey(25)
                cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)

                counter += 1

    cap.release()
    cv2.destroyAllWindows()
    break
