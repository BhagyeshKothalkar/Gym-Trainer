from .chat import VisualChain
from .common import flux_image
from .utils import draw_skeleton_on_image, pil_to_base64

with flux_image.imports():
    import numpy as np
    from PIL import Image
    from scipy.spatial.distance import cdist
    from scipy.stats import zscore


class Alignment:
    def enter(self):
        print("[Alignment][__init__] initializing")
        self.ERROR_THRESHOLD = 0.8
        self.MAX_ERROR_FRAMES = 10
        self.FRAME_CLUSTER_GAP = 10
        self.coach = VisualChain()
        self.coach.enter()

    def find_critical_frames(self, u_feats, t_feats):
        print(
            f"[Alignment][find_critical_frames] u_feats shape={u_feats.shape}, t_feats shape={t_feats.shape}"
        )
        u_feats_norm = np.nan_to_num(zscore(u_feats, axis=0))
        t_feats_norm = np.nan_to_num(zscore(t_feats, axis=0))

        print("Synchronizing videos...")
        dist_matrix = cdist(u_feats_norm, t_feats_norm)
        best_match_indices = np.argmin(dist_matrix, axis=1)

        raw_error_candidates = []

        for u_idx, t_idx in enumerate(best_match_indices):
            cost = dist_matrix[u_idx, t_idx]
            if cost > self.ERROR_THRESHOLD:
                raw_error_candidates.append([cost, u_idx, t_idx])

        unique_errors = []

        if raw_error_candidates:
            raw_error_candidates.sort(key=lambda x: x[1])

            current_cluster = [raw_error_candidates[0]]

            for i in range(1, len(raw_error_candidates)):
                curr_frame = raw_error_candidates[i]
                prev_frame = current_cluster[0]

                if (curr_frame[1] - prev_frame[1]) <= self.FRAME_CLUSTER_GAP:
                    current_cluster.append(curr_frame)
                else:
                    worst_frame_in_cluster = max(current_cluster, key=lambda x: x[0])
                    unique_errors.append(worst_frame_in_cluster)
                    current_cluster = [curr_frame]

            if current_cluster:
                worst_frame_in_cluster = max(current_cluster, key=lambda x: x[0])
                unique_errors.append(worst_frame_in_cluster)

        unique_errors.sort(key=lambda x: x[0], reverse=True)
        top_errors = unique_errors[: self.MAX_ERROR_FRAMES]

        print(f"Found {len(raw_error_candidates)} total bad frames.")
        print(f"Condensed into {len(unique_errors)} distinct error events.")
        print(f"Analyzing top {len(top_errors)} events.")
        return top_errors

    def feedback(self, top_errors, exercise_name, u_images, t_images, u_poses, t_poses):
        print(
            f"[Alignment][feedback] processing {len(top_errors)} top errors for exercise: {exercise_name}"
        )

        print("Generating feedback")
        results = []

        for i, item in enumerate(top_errors):
            u_idx = item[1]
            t_idx = item[2]

            u_img_pil = u_images[u_idx]
            t_img_pil = t_images[t_idx]

            u_img_np = np.array(u_img_pil)
            t_img_np = np.array(t_img_pil)

            u_pose = u_poses[u_idx][0] if u_poses[u_idx] else None
            t_pose = t_poses[t_idx][0] if t_poses[t_idx] else None

            if u_pose:
                u_img_np = draw_skeleton_on_image(u_img_np, u_pose)
            if t_pose:
                t_img_np = draw_skeleton_on_image(t_img_np, t_pose)

            u_img_final = Image.fromarray(u_img_np)
            t_img_final = Image.fromarray(t_img_np)

            u_b64 = pil_to_base64(u_img_final)
            t_b64 = pil_to_base64(t_img_final)

            print(
                f"Requesting AI feedback for frame pair (User: {u_idx}, Trainer: {t_idx})..."
            )
            if i == 0:
                ai_feedback = self.coach.analyze_images(u_b64, t_b64, exercise_name)

            else:
                ai_feedback = "never mind some oher day"

            results.append(
                {
                    "frame_id": int(u_idx),
                    "error_score": round(item[1], 2),
                    "feedback": ai_feedback,
                    "user_image": u_b64,
                    "trainer_image": t_b64,
                }
            )
        return results
