import cv2
def knapsack(shot_scores, shot_change_points, keep):
        durations = [end - start for start, end in shot_change_points]
        total_duration = shot_change_points[-1][1] + 1
        capacity = int(keep * total_duration)
        n = len(durations)
        K = [[0 for x in range(capacity + 1)] for x in range(n + 1)] 
        for i in range(n + 1): 
            for w in range(capacity + 1): 
                if i == 0 or w == 0:
                    K[i][w] = 0 
                elif durations[i-1] <= w: 
                    K[i][w] = max(shot_scores[i-1] + K[i-1][w-durations[i-1]], K[i-1][w]) 
                else: 
                    K[i][w] = K[i-1][w]

        selected = []
        w = capacity
        for i in range(n,0,-1):
            if K[i][w]!= K[i-1][w]:
                selected.insert(0,i-1)
                w -= durations[i-1]
        return selected 

def summary_generator(input_video_path, shot_scores, shot_change_points, output_video_path, keep = 0.15):
    selected_shots = knapsack(shot_scores, shot_change_points, keep = keep)
    cap = cv2.VideoCapture(input_video_path)
    # Get the properties of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    # Iterate through the frame indices
    for shot_index, shot in enumerate(shot_change_points):
        # if selected_shots[shot_index] == 0:
        if shot_index in selected_shots:
            continue
        for index, frame_index in enumerate(start=shot[0], stop=shot[1]+1):
            # Set the frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            # Read the frame
            ret, frame = cap.read()

            # Check if the frame is captured successfully
            if ret:
                # Write the frame to the output video
                out.write(frame)
                print(f"Frame {index} captured.")
            else:
                print(f"Error capturing frame {index}.")

    # Release video capture and writer objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()