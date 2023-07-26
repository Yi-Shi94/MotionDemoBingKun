import numpy as np

class FPS:
    def __init__(self, pcd_xyz, n_samples):
        self.n_samples = n_samples
        self.pcd_xyz = pcd_xyz
        self.n_pts = pcd_xyz.shape[0]
        self.dim = pcd_xyz.shape[1]
        self.selected_pts = None
        self.selected_pts_expanded = np.zeros(shape=(n_samples, 1, self.dim))
        self.remaining_pts = np.copy(pcd_xyz)

        self.grouping_radius = None
        self.dist_pts_to_selected = None
        self.labels = None

        # Random pick a start
        self.start_idx = np.random.randint(low=0, high=self.n_pts - 1)
        self.selected_pts_expanded[0] = self.remaining_pts[self.start_idx]
        self.n_selected_pts = 1

    def get_selected_pts(self):
        self.selected_pts = np.squeeze(self.selected_pts_expanded, axis=1)
        return self.selected_pts

    def step(self):
        if self.n_selected_pts < self.n_samples:
            self.dist_pts_to_selected = self.__distance__(self.remaining_pts, self.selected_pts_expanded[:self.n_selected_pts]).T
            dist_pts_to_selected_min = np.min(self.dist_pts_to_selected, axis=1, keepdims=True)
            res_selected_idx = np.argmax(dist_pts_to_selected_min)
            self.selected_pts_expanded[self.n_selected_pts] = self.remaining_pts[res_selected_idx]

            self.n_selected_pts += 1
        else:
            print("Got enough number samples")


    def fit(self):
        for _ in range(1, self.n_samples):
            self.step()
        return self.get_selected_pts()


    def group(self, radius):
        self.grouping_radius = radius 
        dists = self.dist_pts_to_selected

        dists = np.where(dists > radius, dists+1000000*radius, dists)

        self.labels = np.argmin(dists, axis=1)
        return self.labels


    @staticmethod
    def __distance__(a, b):
        return np.linalg.norm(a - b, ord=2, axis=2)