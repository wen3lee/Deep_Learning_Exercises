import cv2
import argparse
import numpy as np
import time

def main(args):
    # load model
    print("loading model...")
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    # use cuda
    if args.device == 'NvidiaGPU':
        print("use cuda...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    elif args.device == 'IntelGPU':
        print("use Intel gpu...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
    elif args.device == 'VPU':
        print("use vpu...")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
    elif args.device == 'InferenceCPU':
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # load image
    # image from cam
    if args.input_path == 'cam':
        cap = cv2.VideoCapture(0)
    elif args.input_path == 'video':
        cap = cv2.VideoCapture(args.file_path)

    counter = 0
    start_ime=time.time()

    if args.input_path == 'cam' or args.input_path == 'video':
        w = int(cap.get(3))
        h = int(cap.get(4))

        out_video = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h), True)

        start_time=time.time()

        while cap.isOpened():
            flag, frame = cap.read()
            if not flag:
                break

            counter += 1

            # Preprocess image
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))

            # Process image
            net.setInput(blob)
            faces = net.forward()

            # Analyze detections
            for i in range(faces.shape[2]):
                    confidence = faces[0, 0, i, 2]
                    if confidence > args.threshold:
                        box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (x, y, x1, y1) = box.astype("int")
                        cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

            # Output image
            if args.input_path == 'cam' or args.input_path == 'video':
                cv2.imshow('output', frame)
                cv2.waitKey(10)
                out_video.write(frame)

        cap.release()

        total_time = time.time() - start_time
        total_inference_time = round(total_time, 1)
        print("total_time: {}, FPS: {}".format(total_time, counter/total_time))

    # image from pics
    else:
        img = cv2.imread(args.file_path)
        #img = cv2.imread("img-0001.bmp")
        img = cv2.resize(img,(int(500),int(500)))
        h, w = img.shape[:2]
        #print("height:{}, weight:{}".format(h,w))

        # Preprocess image
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))

        # Process image
        net.setInput(blob)
        faces = net.forward()

        # Analyze detections
        for i in range(faces.shape[2]):
                confidence = faces[0, 0, i, 2]
                if confidence > args.threshold:
                    box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x, y, x1, y1) = box.astype("int")
                    cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)

        # Output image
        cv2.imwrite('output.jpg', img);

    cv2.destroyAllWindows()

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_path", required=True, help="cam, video, picture")
    parser.add_argument("-f", "--file_path", default="guitar.mp4")
    parser.add_argument("-d", "--device", default="CPU", help="CPU, InferenceCPU, IntelGPU, NvidiaGPU, VPU")
    parser.add_argument("-t", "--threshold", default=0.5)
    

    args=parser.parse_args()

    main(args)
