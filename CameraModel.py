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
        self.ray_length = 20  # Length of projection rays for visualization

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
        self.extrinsic = self.extrinisic@translation_matrix


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
        direction_x = (img_x - self.cx) * self.mx  # X direction
        direction_y = (img_y - self.cy) * self.my  # Y direction
        direction_z = self.focal_length  # Z direction (camera optical axis)

        # Normalize to create a unit vector
        direction_vector = np.array([direction_x, direction_y, direction_z])
        unit_vector = direction_vector / np.linalg.norm(direction_vector)

        # Compute the ray length based on Z scaling
        self.ray_length = self.z / unit_vector[2]

        # Start and end points of the ray
        start_point = np.array([0, 0, 0])  # Camera center
        end_point = start_point + unit_vector * self.ray_length

        return start_point, end_point

    def get_img_plane(self):
        """
        Get the corners of the image plane in 3D space.

        Returns:
            tuple: Three arrays (img_plane_x, img_plane_y, img_plane_z) for plotting the image plane.
        """
        img_plane_x = (np.array([0, 0, self.img_width, self.img_width]) - self.cx) * self.mx
        img_plane_y = (np.array([0, self.img_height, self.img_height, 0]) - self.cy) * self.my
        img_plane_z = np.full(img_plane_x.shape, self.focal_length)
        return img_plane_x, img_plane_y, img_plane_z

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

    camera_model = CameraModel(camera_parameter)

    img_features = [
        (x, y) for x in range(0, img_width, 128) for y in range(0, img_height, 128)
    ]

    projected_features = [camera_model.project(img_feature) for img_feature in img_features]

    # Plot 3D projections
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("3D Projection of Image Features")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot camera center
    camera_center = projected_features[0][0]
    ax.scatter(*camera_center, color='red', marker='o', label="Camera Center")

    # Plot rays from camera to projected points
    for start_point, end_point in projected_features:
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


