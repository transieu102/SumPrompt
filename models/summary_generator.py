from ortools.algorithms.python import knapsack_solver
import cv2
def knapsack(shot_scores, shot_change_points, keep):
        solver = knapsack_solver.KnapsackSolver(
            knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
            "KnapsackExample",)
        
        durations = [end - start for start, end in shot_change_points]
        total_duration = shot_change_points[-1][1] + 1
        capacity = int(keep * total_duration)
        
        solver.init(shot_scores, durations, [capacity])
        computed_value = solver.solve()

        # Extract which shots are selected
        selected_shots = [0] * len(shot_scores)
        selected_items = solver.best_solution_items()
        for item in selected_items:
            selected_shots[item] = 1

        return selected_shots

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
        if selected_shots[shot_index] == 0:
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