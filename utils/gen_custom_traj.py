import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy import interpolate
from sklearn.decomposition import PCA 


def gen_circle(radius, num_points, R):
  ps = np.arange(num_points)
  pts = (np.exp(2j*np.pi/num_points)**ps*radius)

  transfromations = []
  for x, z in zip(pts.real, pts.imag):
    t = np.array([x, 0, z])

    transformation = np.zeros((3,4))
    transformation[:, :3] = R
    transformation[:, 3] = t

    transfromations += [transformation]

  return np.array(transfromations)


def gen_line(poses, N):
  # R
  slerp = Slerp(np.linspace(1, poses.shape[0], poses.shape[0]), R.from_matrix(poses[:, :3, :3]))
  interp_rots = slerp(np.linspace(1, poses.shape[0], N))

  # T
  tck, u = interpolate.splprep([poses[:, 0, 3], poses[:, 1, 3], poses[:, 2, 3]], s=0.0)
  interp_posits = interpolate.splev(np.linspace(0, 1, N),  tck)

  interp_poses = np.zeros((N, 3, 4))
  for pose_i in range(N):
    interp_poses[pose_i, :3, :3] = interp_rots.as_matrix()[pose_i]
    interp_poses[pose_i, :3, 3] = np.array([interp_posits[0][pose_i], interp_posits[1][pose_i], interp_posits[2][pose_i]])

  return interp_poses


def find_closest_to_center_point(pcloud, center_point):
  min = 0
  id = -1
  for i, point in enumerate(pcloud):
    d = np.linalg.norm(point - center_point)
    if id != -1:
      if d < min:
        min = d
        id = i
    else:
      id = i
      min = d
  return id, min


def gen_grid(poses, R, H, W, id_nerf = None, scale_factor = 0.9, H2W_ratio = 1):
  pcloud_train = poses[..., 3]
  if id_nerf is None:
    center_train_traj = poses[..., 3].mean(0)
    id_train, _ = find_closest_to_center_point(pcloud_train, center_train_traj)
  else:
    id_train = id_nerf

  pca = PCA(n_components=2)
  pca.fit(pcloud_train)

  w_grid = pca.explained_variance_ratio_[0] * scale_factor
  h_grid = w_grid * H2W_ratio

  x = np.linspace(- w_grid, w_grid, W)
  y = np.linspace(- h_grid, h_grid, H)
  x, y = np.meshgrid(x, y)
  x_point = x.flatten()
  y_point = y.flatten()

  poses_val_T = np.vstack((x_point, np.zeros(W * H), y_point)).T
  poses_val = np.zeros((W * H, 3, 4))
  poses_val[..., 3] = poses_val_T
  poses_val[:, :3, :3] = R

  center_val_traj = poses_val[..., 3].mean(0)
  pcloud_val = poses_val[..., 3]
  id_val, _ = find_closest_to_center_point(pcloud_val, center_val_traj)

  delta = pcloud_val[id_val] - pcloud_train[id_train]
  poses_val[:, :, 3] = poses_val[:, :, 3] - delta

  center_val_traj_ = poses_val[..., 3].mean(0)
  id_val_, _ = find_closest_to_center_point(poses_val[..., 3], center_val_traj_)

  print("id_val_: ", id_val_)
  print("id_train: ", id_train)
  
  if not np.array_equal(poses_val[id_val_], poses[id_train]):
    print("!!!Errors in grid generation!!!")

  return poses_val
