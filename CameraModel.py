import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CameraModel:
    """
    A class to represent a pinhole camera model and perform 3D projections.
    """

    def __init__(self, camera_parameter):
        """
        Initialize the CameraModel with intrinsic parameters.
        """
        self.z = 0.1  # Default projection plane depth
        self.depth = 20  # Length of projection rays for visualization

        # Camera intrinsic parameters
        self.cx = camera_parameter["cx"]
        self.cy = camera_parameter["cy"]
        self.fx = camera_parameter["fx"]
        self.fy = camera_parameter["fy"]
        self.mx = camera_parameter["mx"]  # Pixel size in X direction
        self.my = camera_parameter["my"]  # Pixel size in Y direction
        self.focal_length = camera_parameter["focal_length"]
        self.intrinsic = camera_parameter["instrinsic"]


        # Camera exntrinsic parameters
        default_extrinsic = np.array([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
        ])

        self.extrinsic = default_extrinsic

        # Image dimensions
        self.img_width = camera_parameter["img_width"]
        self.img_height = camera_parameter["img_height"]

    def transform(self, transform_matrix):
        """
        Apply transform to the camera extrinsic matrix.

        Parameters:
            transform_matrix (np.ndarray): A 3x3-rotation matrix
        """

        # Apply translation
        self.extrinsic = self.extrinsic@transform_matrix


    def rotate(self, rotation_matrix):
        """
        Apply rotation (adjust orientation) to the camera extrinsic matrix.
    
        Parameters:
            rotation_matrix (np.ndarray): A 3x3 rotation matrix.
        """
        # Create homogeneous rotation matrix
        rotation_homogeneous = np.eye(4)
        rotation_homogeneous[:3, :3] = rotation_matrix
    
        # Apply rotation (update orientation only)
        self.extrinsic[:3, :3] = rotation_matrix @ self.extrinsic[:3, :3]

    def translate(self, translation_vector):
        """
        Apply translation (movement) to the camera extrinsic matrix.

        Parameters:
            translation_vector (np.ndarray): A 3-element array (tx, ty, tz).
        """
        # Create translation matrix
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = translation_vector

        # Apply translation
        self.extrinsic = self.extrinsic@translation_matrix

    '''
    def project(self, img_feature):
        """
        Project a 2D image feature into 3D space using a ray direction vector.

        Parameters:
            img_feature (tuple): A tuple of (img_x, img_y) in image coordinates.

        Returns:
            tuple: A tuple of (start_point, end_point) in 3D space.
        """
        img_x, img_y = img_feature

        # Compute ray direction in 3D space
        direction_x = (img_x - self.cx + 0.5) * self.mx  # X direction # 0.5 is pixel center
        direction_y = (img_y - self.cy + 0.5) * self.my  # Y direction # 0.5 is pixel center

        direction_z = self.focal_length  # Z direction (camera optical axis)

        # Normalize to create a unit vector
        direction_vector = np.array([direction_x, direction_y, direction_z])
        unit_vector = direction_vector / np.linalg.norm(direction_vector)

        # Compute the ray length based on Z scaling
        self.depth = self.z / unit_vector[2]

        # Start and end points of the ray
        start_point = np.array([0, 0, 0])  # Camera center
        end_point = start_point + unit_vector * self.depth

        return start_point, end_point
    '''

    def project_camera(self, img_features):
        """
        Project 2D image features into 3D rays in the camera coordinate system.

        Parameters:
            img_features (list): A list of img_feature;(img_x, img_y) in image coordinates.

        Returns:
            np.array: A ray array in 3D space.
        """

        #img_features = np.atleast_2d(img_features)  # Ensure 2D array
        img_x = img_features[:, 0]
        img_y = img_features[:, 1]
    
        # Compute ray direction
        direction_x = (img_x - self.cx + 0.5) * self.mx
        direction_y = (img_y - self.cy + 0.5) * self.my
        direction_z = np.full_like(direction_x, self.focal_length)
    
        # Normalize direction vectors
        directions = np.vstack([direction_x, direction_y, direction_z]).T
        unit_vectors = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
    
        # Compute start and end points
        start_point = np.array([0,0,0])
        depths = self.z / unit_vectors[:, 2]
        end_points = start_point + unit_vectors * depths[:, np.newaxis]
    
        # Return as list of tuples
        return (start_point, end_points)

    def project(self, img_features):
        (start_point, end_points) = self.project_camera(img_features)
        
        # Transform points from camera space to world space
        start_point_h = np.append(start_point, 1)  # Homogeneous coordinate
        end_points_h = np.hstack([end_points, np.ones((end_points.shape[0], 1))])  # Homogeneous

        # Apply transformation matrix
        start_point_w = (self.extrinsic @ start_point_h.T).T[:3]
        end_points_w  = (self.extrinsic @ end_points_h.T).T[:, :3]

        return (start_point_w, end_points_w)




    def get_img_plane_camera(self, img_shape, focal_length = 0.1, principal_point = (960.5, 540.5), sensor_size = (0.1, 0.1)):
        """
        Create the corners of the image plane in 3D space.
    
        Args:
            image_size: (width, height) 형태의 이미지 크기 튜플
            focal_length: 초점 거리
            principal_point: (cx, cy) 형태의 이미지 평면상의 가로 세로 주점의 위치
            sensor_size: (mx, my) 픽셀당 이미지 센서의 실제 가로 세로 크기
    
        Returns:
            numpy.ndarray: 3x(width*height) 형태의 이미지 평면 좌표
        """
        img_width, img_height = img_shape
        cx, cy = principal_point
        mx, my = sensor_size
        x = (np.array([0, 0, img_width, img_width]) - cx) * mx
        y = (np.array([0, img_height, img_height, 0]) - cy) * my
        z = np.ones_like(x) * focal_length
    
        # 3x(width*height) 형태로 변환
        img_plane = np.vstack((x.flatten(), y.flatten(), z.flatten()))
        return img_plane

    def get_img_plane(self):
        img_plane = self.get_img_plane_camera((self.img_width, self.img_height), focal_length = self.focal_length, 
            principal_point = (self.cx, self.cy), sensor_size = (self.mx, self.my))
        
        ones = np.ones((1, img_plane.shape[1]))
        img_plane_homogeneous = np.vstack((img_plane, ones))

        # Extrinsic 행렬을 적용 (4x4 @ 4xN → 4xN)
        transformed_img_plane = self.extrinsic @ img_plane_homogeneous

        # 동차좌표계 → 3D 좌표계로 변환 (4xN → 3xN)
        transformed_img_plane_3d = transformed_img_plane[:3, :] / transformed_img_plane[3, :]

        return (transformed_img_plane_3d[0,:], transformed_img_plane_3d[1,:], transformed_img_plane_3d[2,:])

def main():
    # Camera intrinsic parameters
    img_width = 1920
    img_height = 1080
    focal_length = 0.1
    mx, my = 0.01, 0.01  # Pixel size in real world
    fx = focal_length / mx
    fy = focal_length / my
    cx = (img_width - 1) / 2
    cy = (img_height - 1) / 2

    # Camera intrinsic matrix
    camera_instrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ])

    # Camera parameter dictionary
    camera_parameter = {
        "img_width": img_width,
        "img_height": img_height,
        "focal_length": focal_length,
        "fx": fx,
        "fy": fy,
        "mx": mx,
        "my": my,
        "cx": cx,
        "cy": cy,
        "instrinsic": camera_instrinsic,
    }

    # Plot 3D projections
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Projection of Image Features")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


    camera_model = CameraModel(camera_parameter)

    img_features = np.array([
        [x, y] for x in range(0, img_width, 128) for y in range(0, img_height, 128)
    ])

    projected_features = camera_model.project(img_features)
    (start_point, end_points) = projected_features

    # Plot camera center
    camera_center = start_point
    ax.scatter(*camera_center, color='red', marker='o', label="Camera Center")


    # Plot rays from camera to projected points
    for end_point in end_points:
        ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]], 'g--')
        ax.scatter(*end_point, color='blue', marker='x')

    # Plot image plane
    img_plane_x, img_plane_y, img_plane_z = camera_model.get_img_plane()

    ax.plot_trisurf(img_plane_x, img_plane_y, img_plane_z, color='cyan', alpha=0.3)
    ax.text(img_plane_x.mean(), img_plane_y.mean(), focal_length, "Image Plane", color='cyan')

    # Show plot
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()


