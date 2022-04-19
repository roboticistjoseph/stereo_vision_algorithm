
"""
    Importing Necessary Libraries
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_fundamental_matrix(features):
    """
    The fundamental matrix is computed and the outliers are eliminated, keeping only the best set of inliers

    :param p1: matches obtanied from filtering BF matcher output

    :return: Fundamental Matrix and Inliers
    """ 
    iterations = 1000
    inliers_thresh = 0
    chosen_indices = []
    f_matrix = 0

    for i in range(0, iterations):
        indices = []
        n_rows = features.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        features_8 = features[random_indices, :] 
        
        normalised = True

        x1 = features_8 [:,0:2]
        x2 = features_8 [:,2:4]

        if x1.shape[0] > 7:
            if normalised == True:
                x1_norm, T1 = normalize(x1)
                x2_norm, T2 = normalize(x2)
            else:
                x1_norm,x2_norm = x1,x2
                
            A = np.zeros((len(x1_norm),9))
            for i in range(0, len(x1_norm)):
                x_1,y_1 = x1_norm[i][0], x1_norm[i][1]
                x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
                A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

            U, S, VT = np.linalg.svd(A, full_matrices=True)
            F = VT.T[:, -1]
            F = F.reshape(3,3)

            u, s, vt = np.linalg.svd(F)
            s = np.diag(s)
            s[2,2] = 0
            F = np.dot(u, np.dot(s, vt))

            if normalised:
                F = np.dot(T2.T, np.dot(F, T1))
                f_8 = F

        else:
            f_8 = None
        
        for j in range(n_rows):
            feature = features[j]
            x1,x2 = feature[0:2], feature[2:4]
            x1tmp=np.array([x1[0], x1[1], 1]).T
            x2tmp=np.array([x2[0], x2[1], 1])

            error = np.dot(x1tmp, np.dot(f_8, x2tmp))
            error = np.abs(error)
            
            
            # error = errorF(feature, f_8)
            if error < 0.02:
                indices.append(j)

        if len(indices) > inliers_thresh:
            inliers_thresh = len(indices)
            chosen_indices = indices
            f_matrix = f_8

    filtered_features = features[chosen_indices, :]
    return f_matrix, filtered_features

def compute_essential_matrix(K1, K2, F):
    """
    Computing Essential Matrix

    :param p1: Camera 1 Matrix
    :param p2: Camera 2 Matrix

    :return: essential matrix
    """ 
    E = K2.T.dot(F).dot(K1)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    essential_matrix = np.dot(U,np.dot(np.diag(s),V))
    return essential_matrix


    
def compute_camera_pose(E):
    """
    Computing the pose of the camera when it was tking the image

    :param p1: essential matrix

    :return: rotational and translational matrix of camera
    """ 
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C

def cheirality_condition(P_3D,C_,R_3):
    """
    This condition checks if the desired pixel is in front of the camera

    :param p1: points after triangulation
    :param p2: translational matrix
    :param p3: rotational mmatrix
    
    :return: boolean vaule saying if the condition is satisfied or not
    """ 
    num_positive = 0
    for P in P_3D:
        P = P.reshape(-1,1)
        if R_3.dot(P - C_) > 0 and P[2]>0:
            num_positive+=1
    return num_positive


def normalize(data):

    data_dash = np.mean(data, axis=0)
    u_dash ,v_dash = data_dash[0], data_dash[1]

    u_cap = data[:,0] - u_dash
    v_cap = data[:,1] - v_dash

    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    T_scale = np.diag([s,s,1])
    T_trans = np.array([[1,0,-u_dash],[0,1,-v_dash],[0,0,1]])
    T = T_scale.dot(T_trans)

    x_ = np.column_stack((data, np.ones(len(data))))
    x_norm = (T.dot(x_.T)).T

    return  x_norm, T


"""
    Main Function
""" 
if __name__ == '__main__':
    
    """
        Choose the desired Dataset Number
    """
    dataset_number = 3
    
    if dataset_number == 1:
        dataset = 'Curule Dataset'
        img1 = cv2.imread('data/curule/im0.png')
        img2 = cv2.imread('data/curule/im1.png')
        K1 = np.array([[1758.23, 0, 977.42],[ 0, 1758.23, 552.15],[ 0, 0, 1]])
        K2 = np.array([[1758.23, 0, 977.42],[ 0, 1758.23, 552.15],[ 0, 0, 1]])
        baseline=88.39
        f = K1[0,0]

    elif dataset_number == 2:
        dataset = 'Octagon Dataset'
        img1 = cv2.imread('data/octagon/im0.png')
        img2 = cv2.imread('data/octagon/im1.png')
        K1 = np.array([[1742.11, 0, 804.90],[ 0, 1742.11, 541.22],[ 0, 0, 1]])
        K2 = np.array([[1742.11, 0, 804.90],[ 0, 1742.11, 541.22],[ 0, 0, 1]])
        baseline=221.76
        f = K1[0,0]
        
    elif dataset_number == 3:
        dataset = 'Pendulum Dataset'
        img1 = cv2.imread('data/pendulum/im0.png')
        img2 = cv2.imread('data/pendulum/im1.png')
        K1 = np.array([[1729.05, 0, -364.24],[ 0, 1729.05, 552.22],[ 0, 0, 1]])
        K2 = np.array([[1729.05, 0, -364.24],[ 0, 1729.05, 552.22],[ 0, 0, 1]])
        baseline=537.75
        f = K1[0,0]

    else:
        print("Data-set does not exist")
    
    
    print('[INFO]... Chosen Dataset is: ', dataset)
    print('Camera Matrix= \n', K1)
    print('Baseline= ', baseline, '\n')
    
    """
        Pre-processing Images
    """
    # Creating a Copy of Input Images
    print('[INFO]... Reading Input Images')
    image0 = img1
    image1 = img2
    
    # Converting to Gray Scale Images
    print('[INFO]... Converting input Images to Gray Scale')
    gray_image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY) 
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    
    # Converting to RGB Image
    print('[INFO]... Creating a RGB copy of input Images')
    rgb_image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB) 
    rgb_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
  
    """
        Calibration
    """ 
    
    # Finding Key points using SIFT Detector
    print('[INFO]... Finding Key points using SIFT Detector')
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_image0, None)
    kp2, des2 = sift.detectAndCompute(gray_image1, None)
    
    # Finding Matching points
    print('[INFO]... Finding Matching points')
    bf = cv2.BFMatcher()
    matches = bf.match(des1,des2)
    
    # Filtering the Matching Points
    matches = sorted(matches, key = lambda x :x.distance)
    filtered_matches = matches[0:100]
    
    # Draw lines between the similar Matching points of given tow images
    matched_image = cv2.drawMatches(rgb_image0,kp1,rgb_image1,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(matched_image)
    plt.savefig('features_image.png')
    
    # Converting the obtained Matches into an Array
    matched_pairs = []
    for i, m1 in enumerate(filtered_matches):
        pt1 = kp1[m1.queryIdx].pt
        pt2 = kp2[m1.trainIdx].pt
        matched_pairs.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    matched_pairs = np.array(matched_pairs).reshape(-1, 4)
    
    # Get Fundamental Matrix
    print('[INFO]... Finding Fundamental Matrix')
    fundamental_matrix, matched_pairs_inliers = compute_fundamental_matrix(matched_pairs)
    
    # Get Essential Matrix
    print('[INFO]... Finding Essential Matrix')
    essential_matrix = compute_essential_matrix(K1, K2, fundamental_matrix)
    
    # Get Camera Pose
    print('[INFO]... Finding Camera Pose')
    init_rotational_matrix, init_translational_matrix = compute_camera_pose(essential_matrix)
    
    # Projection Matrix
    Pts_3D = []
    R1  = np.identity(3)
    C1  = np.zeros((3, 1))
    I = np.identity(3)
    for i in range(len(init_rotational_matrix)):
        R2 =  init_rotational_matrix[i]
        C2 =   init_translational_matrix[i].reshape(3,1)
        ProjectionM_left = np.dot(K1, np.dot(R1, np.hstack((I, -C1.reshape(3,1)))))
        ProjectionM_right = np.dot(K2, np.dot(R2, np.hstack((I, -C2.reshape(3,1)))))
        
        # Triangulation
        for xL,xR in zip(matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]):
            pts_3d = cv2.triangulatePoints(ProjectionM_left, ProjectionM_right, np.float32(xL), np.float32(xR))
            pts_3d = np.array(pts_3d)
            pts_3d = pts_3d[0:3,0]
            Pts_3D.append(pts_3d)

    # Finding Actual camera pose from the obtained 4 sets of values
    best_i = 0
    max_Positive = 0
    for i in range(len(init_rotational_matrix)):
        R_, C_ = init_rotational_matrix[i],  init_translational_matrix[i].reshape(-1,1)
        R_3 = R_[2].reshape(1,-1)
        num_Positive = cheirality_condition(Pts_3D,C_,R_3)

        if num_Positive > max_Positive:
            best_i = i
            max_Positive = num_Positive

    R_Config, C_Config, P3D = init_rotational_matrix[best_i], init_translational_matrix[best_i], Pts_3D[best_i]
    
    print('The Funndamental Matrix: \n', fundamental_matrix,'\n')
    print('The Essential Matrix: \n', essential_matrix,'\n')
    print('The Rotation Matrix: \n', R_Config,'\n')
    print('The Translation Matrix: \n', C_Config, '\n')
    
    
    """
        Rectification
    """
    
    # Matched pairs in Image 1 and Image 2
    set1, set2 = matched_pairs_inliers[:,0:2], matched_pairs_inliers[:,2:4]
        
    h1, w1 = image0.shape[:2]
    h2, w2 = image1.shape[:2]
    
    print('[INFO]... Computing Homography Matrices \n')
    # Homographic Transformation to project the image plane of each camera onto a perfectly aligned virtual plane
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(set1), np.float32(set2), fundamental_matrix, imgSize=(w1, h1))
    print("Estimated H1 and H2 as \n Homography Matrix 1: \n", H1,'\nHomography Matrix 2:\n ', H2)
    
    print('[INFO]... Warping the Image \n')
    # Warping image to make the epipolar lines parallel
    img1_rectified = cv2.warpPerspective(image0, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(image1, H2, (w2, h2))
    set1_rectified = cv2.perspectiveTransform(set1.reshape(-1, 1, 2), H1).reshape(-1,2)
    set2_rectified = cv2.perspectiveTransform(set2.reshape(-1, 1, 2), H2).reshape(-1,2)

    # Reshaping the rectified Image
    img1_rectified_reshaped = cv2.resize(img1_rectified, (int(img1_rectified.shape[1] / 4), int(img1_rectified.shape[0] / 4)))
    img2_rectified_reshaped = cv2.resize(img2_rectified, (int(img2_rectified.shape[1] / 4), int(img2_rectified.shape[0] / 4)))

    # Gray scaling the Rectified Image
    img1_rectified_reshaped = cv2.cvtColor(img1_rectified_reshaped, cv2.COLOR_BGR2GRAY)
    img2_rectified_reshaped = cv2.cvtColor(img2_rectified_reshaped, cv2.COLOR_BGR2GRAY)
    
    """
        Correspondence
    """
  
    left_array, right_array = img1_rectified_reshaped, img2_rectified_reshaped
    left_array = left_array.astype(int)
    right_array = right_array.astype(int)
    if left_array.shape != right_array.shape:
        raise "Inequal Image Shape"
    h, w = left_array.shape
    
    # After a lot of trial and error, settled in on these values for each dataset
    if(dataset_number == 1):
        depth_thresh = 90000
        window = 5
    elif(dataset_number == 2):
        depth_thresh = 1000000
        window = 7
    elif(dataset_number == 3):
        depth_thresh = 100000
        window = 3
        
    print('[INFO]... Computing Disparity Map')
    disparity_map = np.zeros((h, w))
    x_new = w - (2 * window)
    for y in range(window, h-window):
        block_left_array = []
        block_right_array = []
        for x in range(window, w-window):
            block_left = left_array[y:y + window, x:x + window]
            block_left_array.append(block_left.flatten())
  
            block_right = right_array[y:y + window, x:x + window]
            block_right_array.append(block_right.flatten())
  
        block_left_array = np.array(block_left_array)
        block_left_array = np.repeat(block_left_array[:, :, np.newaxis], x_new, axis=2)
  
        block_right_array = np.array(block_right_array)
        block_right_array = np.repeat(block_right_array[:, :, np.newaxis], x_new, axis=2)
        block_right_array = block_right_array.T
  
        abs_diff = np.abs(block_left_array - block_right_array)
        sum_abs_diff = np.sum(abs_diff, axis = 1)
        idx = np.argmin(sum_abs_diff, axis = 0)
        disparity = np.abs(idx - np.linspace(0, x_new, x_new, dtype=int)).reshape(1, x_new)
        disparity_map[y, 0:x_new] = disparity 
  
    # Final disparity Map
    disparity_map_int = np.uint8(disparity_map * 255 / np.max(disparity_map))

#--------------------DEPTH--------------------------------------------------
    """
        Depth
    """
    print('[INFO]... Computing Depth Map')
    
    if(dataset_number == 1):
        depth = (baseline * f) / (disparity_map + 1e-10)
        depth[depth > depth_thresh] = depth_thresh
    elif(dataset_number == 2):
        depth = (baseline * f) / (disparity_map + 1e-10)
        depth[depth > depth_thresh] = depth_thresh
    elif(dataset_number == 3):
        depth = (baseline * f) / (disparity_map + 1e-10)
        depth[depth > depth_thresh] = depth_thresh

    # Final Depth Map
    depth_map = np.uint8(depth * 255 / np.max(depth))
    
    # Saving all the outputs
    plt.imshow(disparity_map_int, cmap='hot', interpolation='nearest')
    plt.savefig('disparity_heat_map' +str(dataset_number)+ ".png")
    plt.imshow(disparity_map_int, cmap='gray', interpolation='nearest')
    plt.savefig('disparity_gray_image' +str(dataset_number)+ ".png")
    plt.imshow(depth_map, cmap='hot', interpolation='nearest')
    plt.savefig('depth_heat_map' +str(dataset_number)+ ".png")
    plt.imshow(depth_map, cmap='gray', interpolation='nearest')
    plt.savefig('depth_gray_image' +str(dataset_number)+ ".png")