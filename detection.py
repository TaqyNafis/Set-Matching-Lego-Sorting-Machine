import cv2
import os
from sorting_pipeline import classify_and_vote

def run_detection():
    # Settings
    VIDEO_PATH = 0
    MIN_AREA = 1500
    WARMUP_FRAMES = 30
    SAVE_FOLDER = "captures"
    TRIGGER_POSITION = 0.7
    PADDING = 10
    BURST_LIMIT = 5

    NO_OBJECT_RESET_FRAMES = 8

    os.makedirs(SAVE_FOLDER, exist_ok=True)

    # Initializationx
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        exit()

    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=80,
        detectShadows=False
    )

    frame_count = 0
    capture_count = 0

    burst_active = False
    burst_counter = 0
    object_processed = False
    burst_image_paths = []

    no_object_frames = 0

    show_trigger_text = False
    trigger_display_counter = 0

    # Main Loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        fgMask = backSub.apply(frame)

        if frame_count <= WARMUP_FRAMES:
            cv2.imshow("Frame", frame)
            cv2.imshow("Mask", fgMask)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        _, thresh = cv2.threshold(fgMask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        trigger_x = int(frame.shape[1] * TRIGGER_POSITION)

        # Uncomment to visualize trigger line
        # cv2.line(frame,
        #          (trigger_x, 0),
        #          (trigger_x, frame.shape[0]),
        #          (255, 0, 0), 2)

        valid_contours = [
            c for c in contours
            if cv2.contourArea(c) > MIN_AREA
        ]

        # Reset when no object present
        if len(valid_contours) == 0:
            no_object_frames += 1

            if no_object_frames >= NO_OBJECT_RESET_FRAMES:
                burst_active = False
                burst_counter = 0
                object_processed = False
                burst_image_paths.clear()

        else:
            no_object_frames = 0

            valid_contours = sorted(
                valid_contours,
                key=lambda c: cv2.boundingRect(c)[0],
                reverse=True
            )

            cnt = valid_contours[0]
            x, y, w, h = cv2.boundingRect(cnt)

            frame_h, frame_w = frame.shape[:2]

            fully_inside = (
                x > 0 and
                y > 0 and
                x + w < frame_w and
                y + h < frame_h
            )

            # Padding
            x_pad = max(0, x - PADDING)
            y_pad = max(0, y - PADDING)
            w_pad = min(frame_w - x_pad, w + 2 * PADDING)
            h_pad = min(frame_h - y_pad, h + 2 * PADDING)

            cv2.rectangle(frame,
                        (x_pad, y_pad),
                        (x_pad + w_pad, y_pad + h_pad),
                        (0, 255, 0), 2)

            # Start Burst Capture (80%)
            if w > 0:
                passed_ratio = max(0, min(1, (trigger_x - x) / w))
            else:
                passed_ratio = 0

            if (
                passed_ratio >= 0.8 and
                fully_inside and
                not burst_active and
                not object_processed
            ):
                burst_active = True
                burst_counter = 0
                burst_image_paths = []
                print("Starting burst capture")

    
            # Burst Capture Logic
            if burst_active:

                # Early exit
                if not fully_inside:

                    print(f"Object left early at {burst_counter} images")

                    burst_active = False
                    object_processed = True

                    if burst_counter > 0:
                        print("Starting early classification...")
                        final_label = classify_and_vote(burst_image_paths)
                        print("FINAL VOTED RESULT:", final_label)

                    burst_image_paths.clear()

                # Continue capturing
                elif burst_counter < BURST_LIMIT:

                    crop = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]

                    filename = os.path.join(
                        SAVE_FOLDER,
                        f"capture_{capture_count}.jpg"
                    )

                    cv2.imwrite(filename, crop)
                    burst_image_paths.append(filename)

                    capture_count += 1
                    burst_counter += 1

                    print(f"Burst image {burst_counter}/{BURST_LIMIT}")

                    show_trigger_text = True
                    trigger_display_counter = 10

                # Normal finish
                else:

                    burst_active = False
                    object_processed = True

                    print("Burst finished. Starting classification...")

                    final_label = classify_and_vote(burst_image_paths)
                    print("FINAL VOTED RESULT:", final_label)

                    burst_image_paths.clear()

        # Display burst counter
        if show_trigger_text:
            cv2.putText(frame,
                        f"BURST {burst_counter}/{BURST_LIMIT}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 0),
                        3)

            trigger_display_counter -= 1
            if trigger_display_counter <= 0:
                show_trigger_text = False

        cv2.imshow("Frame", frame)
        cv2.imshow("Mask", thresh)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_detection()