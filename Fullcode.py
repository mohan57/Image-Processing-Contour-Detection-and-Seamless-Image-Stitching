
1. **Contour Detection**. In this problem we will build a basic contour detector. 

  We have implemented a contour detector that uses the magnitude of the local image gradient as the boundary score as seen below:
"""

from google.colab import drive
drive.mount('/content/drive')

from PIL import Image
import numpy as np
import cv2, os
from scipy.io import loadmat
from scipy import signal
import matplotlib.pyplot as plt
import skimage
import evaluate_boundaries


N_THRESHOLDS = 99

def detect_edges(imlist, fn):
  """
  Detects edges in every image in the image list.
  
  :param imlist: a list of filenames.
  :param fn: an edge detection function.
  
  return: (list of image arrays, list of edge arrays). Edge array is of the same size as the image array.
  """  
  images, edges = [], []
  for imname in imlist:
    I = cv2.imread(os.path.join('/content/drive/MyDrive/pset3/data', str(imname)+'.jpg'))
    images.append(I)

    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = I.astype(np.float32)/255.
    mag = fn(I)
    edges.append(mag)
  return images, edges

def evaluate(imlist, all_predictions):
  """
  Compares the predicted edges with the ground truth.
  
  :param imlist: a list of images.
  :param all_predictions: predicted edges for imlist.
  
  return: the evaluated F1 score.
  """  
  count_r_overall = np.zeros((N_THRESHOLDS,))
  sum_r_overall = np.zeros((N_THRESHOLDS,))
  count_p_overall = np.zeros((N_THRESHOLDS,))
  sum_p_overall = np.zeros((N_THRESHOLDS,))
  for imname, predictions in zip(imlist, all_predictions):
    gt = loadmat(os.path.join('/content/drive/MyDrive/pset3/data', str(imname)+'.mat'))['groundTruth']
    num_gts = gt.shape[1]
    gt = [gt[0,i]['Boundaries'][0,0] for i in range(num_gts)]
    count_r, sum_r, count_p, sum_p, used_thresholds = \
              evaluate_boundaries.evaluate_boundaries_fast(predictions, gt, 
                                                           thresholds=N_THRESHOLDS,
                                                           apply_thinning=True)
    count_r_overall += count_r
    sum_r_overall += sum_r
    count_p_overall += count_p
    sum_p_overall += sum_p

  rec_overall, prec_overall, f1_overall = evaluate_boundaries.compute_rec_prec_f1(
        count_r_overall, sum_r_overall, count_p_overall, sum_p_overall)
  
  return max(f1_overall)

def compute_edges_dxdy(I):
  """
  Returns the norm of dx and dy as the edge response function.
  
  :param I: image array
  
  return: edge array
  """
  
  dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same')
  dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same')
  mag = np.sqrt(dx**2 + dy**2)
  mag = normalize(mag)
  return mag

def normalize(mag):
  """
  Normalizes the edge array to [0,255] for display.
  
  :param mag: unnormalized edge array.
  
  return: normalized edge array
  """  
  mag = mag / 1.5
  mag = mag * 255.
  mag = np.clip(mag, 0, 255)
  mag = mag.astype(np.uint8)
  return mag


imlist = [12084, 24077, 38092]
fn = compute_edges_dxdy
images, edges = detect_edges(imlist, fn)
display(Image.fromarray(np.hstack(images)))
display(Image.fromarray(np.hstack(edges)))
f1 = evaluate(imlist, edges)
print('Overall F1 score:', f1)

"""  **a) Warm-up .** As you visualize the produced edges, you will notice artifacts at image boundaries. Modify how the convolution is being done to minimize these artifacts."""

def compute_edges_dxdy_warmup(I):
    """
    Returns the norm of dx and dy as the edge response function.

    :param I: image array

    return: edge array, which is a HxW numpy array
    """

    # i have take parameters into variables 
    o=1
    z=0
    ddx=5;ddy=5;# this the as same inbuild function 
    k=1# the size of the kernel 
    op=cv2.Sobel#the operator is sobel here
    Isob=I
    dx_kernel = np.array([[-1, 0, 1]])
    dx = signal.convolve2d(I, dx_kernel, mode='same')
    dy = op(Isob, ddy*1, z*0, o*1, ksize=k*1);
    
    mag = np.sqrt(dx ** 2 + dy ** 2)
    mag = normalize(mag)
    return mag



imlist = [12084, 24077, 38092]
fn = compute_edges_dxdy_warmup
images, edges = detect_edges(imlist, fn)
display(Image.fromarray(np.hstack(images)))
display(Image.fromarray(np.hstack(edges)))
f1 = evaluate(imlist, edges)
print('Overall F1 score:', f1)

""" **b) Smoothing  .** Next, notice that we are using [−1, 0, 1] filters for computing the gradients, and they are susceptible to noise. Use derivative of Gaussian filters to obtain more robust estimates of the gradient. Experiment with different sigma for this Gaussian filtering and pick the one that works the best."""

def compute_edges_dxdy_smoothing(I, sigma=1):
    """
    Returns the norm of dx and dy as the edge response function.

    :param I: image array
    :param sigma: standard deviation of Gaussian filter

    return: edge array, which is a HxW numpy array
    """

    o=1;z=0;d=6;
    k=1
    op=cv2.Sobel
    sigma=1.5
    k2=3 #this is the kernel size of Gaussian blr
    igauss = cv2.GaussianBlur(I, (k2*1, k2*1), sigmaX=sigma, sigmaY=sigma)
    dx = op(igauss, d*1, o, z, ksize=k*1);dy = op(igauss, d*1, z*0, o*1, ksize=k*1);

    # compute magnitude of the gradients
    mag = np.sqrt(dx ** 2 + dy ** 2)
    mag = normalize(mag)
    return mag


imlist = [12084, 24077, 38092]
fn = compute_edges_dxdy_smoothing
images, edges = detect_edges(imlist, fn)
display(Image.fromarray(np.hstack(images)))
display(Image.fromarray(np.hstack(edges)))
f1 = evaluate(imlist, edges)
print('Overall F1 score:', f1)

"""**c) Non-maximum Suppression  .** The current code does not produce thin edges. Implement non-maximum suppression, where we look at the gradient magnitude at the two neighbours in the direction perpendicular to the edge. We suppress the output at the current pixel if the output at the current pixel is not more than at the neighbors. You will have to compute the orientation of the contour (using the X and Y gradients), and then lookup values at the neighbouring pixels."""





def compute_edges_dxdy_nonmax(I):
    """
    Returns the norm of dx and dy as the edge response function after non-maximum suppression.

    :param I: image array

    return: edge array, which is a HxW numpy array
    """

    """ Copy over your response from part b and alter it to include this response"""
    # ADD YOUR CODE in part b

    dx = signal.convolve2d(I,np.array([[1,0,-1],[2,0,-2],[1,0,-1]]),mode='same')
    dy = signal.convolve2d(I,np.array([[0]]),mode='same')

    mag = np.sqrt(dx**2 + dy**2)
    mag = normalize(mag)
    #here i am trying to take the kernel shape
    kx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    ky = np.array([[0]])
    og = np.arctan2(signal.convolve2d(I,kx,mode='same'), signal.convolve2d(I,ky,mode='same'))
    suppressed_mag = np.zeros(normalize(mag).shape, dtype=normalize(mag).dtype)
    rw, cl = 1, 1
    one=1
    while rw < mag.shape[0]-one and cl < mag.shape[1]-one:
      try:
        c_og = og[rw, cl]
        if abs(c_og) <= np.pi/4*2 or abs(c_og) >= 7 * np.pi/4*2:
            nb1, nb2 = np.roll(mag[rw, cl-1:cl+2], -1)[1:]
        elif abs(c_og - np.pi/4*1+0) < np.pi/4*2:
            nb1, nb2 = mag[[rw-1,rw+1], [cl+1,cl-1]]
        elif abs(abs(c_og) - np.pi) <= np.pi/4*2+0 or abs(c_og) <= np.pi/4*2+0:
            nb1, nb2 = mag[(rw-1, rw+1), cl]
        else:
            nb1, nb2 = mag[rw - 1 : rw + 2, cl - 1 : cl + 2][[0, 2], [2, 0]]    
        if mag[rw, cl] <= nb1 or mag[rw, cl] <= nb2:
            suppressed_mag[rw, cl] = 0
        else:
            suppressed_mag[rw, cl] = mag[rw, cl]
      except ValueError:
        pass
      cl += 1
      if cl == mag.shape[1]-one:
        rw += 1
        cl = 1
    return suppressed_mag

imlist = [12084, 24077, 38092]
fn = compute_edges_dxdy_nonmax
images, edges = detect_edges(imlist, fn)
display(Image.fromarray(np.hstack(images)))
display(Image.fromarray(np.hstack(edges)))
f1 = evaluate(imlist, edges)
print('Overall F1 score:', f1)

""" 3. **Stitching pairs of images **. In this problem we will estimate homography transforms to register and stitch image pairs. We are providing a image pairs that you should stitch together. We have also provided sample output, though, keep in mind that your output may look different from the reference output depending on implementation details.
 
   **Getting Started** We have implemented some helper functions to detect feature points in both images and extract descriptor of every keypoint in both images. We used SIFT descriptors from OpenCV library. You can refer to this [tutorial](https://docs.opencv.org/4.5.1/da/df5/tutorial_py_sift_intro.html) for more details about using SIFT in OpenCV. Please use opencv version 4.5 or later.
"""

# some helper functions

def imread(fname):
    """
    Read image into np array from file.
    
    :param fname: filename.
    
    return: image array.
    """
    return cv2.imread(fname)

def imread_bw(fname):
    """
    Read image as gray scale format.
    
    :param fname: filename.
    
    return: image array.
    """
    return cv2.cvtColor(imread(fname), cv2.COLOR_BGR2GRAY)

def imshow(img):
    """
    Show image.
    
    :param img: image array.
    """
    skimage.io.imshow(img)
    
def get_sift_data(img):
    """
    Detect the keypoints and compute their SIFT descriptors with opencv library
    
    :param img: image array.
    
    return: (keypoints array, descriptors). keypoints array (Nx2) contains the coordinate (x,y) of each keypoint. 
    Descriptors have size Nx128.
    """
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    kp = np.array([k.pt for k in kp])
    return kp, des

def plot_inlier_matches(ax, img1, img2, inliers):
    """
    Plot the match between two image according to the matched keypoints
    
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :param inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')
    
    ax.plot(inliers[:,2], inliers[:,3], '+r')
    ax.plot(inliers[:,0] + img1.shape[1], inliers[:,1], '+r')
    ax.plot([inliers[:,2], inliers[:,0] + img1.shape[1]],
            [inliers[:,3], inliers[:,1]], 'r', linewidth=0.4)
    ax.axis('off')

"""**a) Putative Matches .** Selectes putative matches based on the matrix of pairwise descriptor distances.  First, Compute distances between every descriptor in one image and every descriptor in the other image. We will use `scipy.spatial.distance.cdist(X, Y, 'sqeuclidean')`. Then, you can select all pairs  whose descriptor distances are below a specified threshold, or select the top few hundred descriptor pairs with the smallest pairwise distances. In your report, display the putative matches overlaid on the image pairs."""

import scipy

def get_best_matches(img1, img2, num_matches):
    """
    Returns the matched keypoints between img1 and img2.
    :param img1: left image.
    :param img2: right image.
    :param num_matches: the number of matches that we want.
    
    return: pixel coordinates of the matched keypoints (x,y in the first image and x,y in the second image), which is a Nx4 numpy array. 
    """
    kp1, des1 = get_sift_data(img1)
    kp2, des2 = get_sift_data(img2)

    # Find distance between descriptors in images
    dist = scipy.spatial.distance.cdist(des1, des2, 'sqeuclidean')
    
    pif = np.argpartition(dist.flatten(), num_matches)[:num_matches]
    sip = np.argsort(dist.flatten()[pif])
    sif = pif[sip]

    rw, cw = dist.shape
    iex1 = []
    iex2 = [] 
    c = 0
    while c < len(sif):
      flattened_index = sif[c]
      row_index = flattened_index // cw
      column_index = flattened_index % cw
      iex1.append(row_index)
      iex2.append(column_index)
      c += 1
    iex1 = np.array(iex1)
    iex2 = np.array(iex2)
    matched_kp1 = []
    matched_kp2 = []  
    c1 = 0
    c2 = 0
    while c1 < len(iex1):

      idx = iex1[c1]
      matched_kp = kp1[idx]
      matched_kp1.append(matched_kp)
      c1 += 1

    while c2 < len(iex2):
      idx = iex2[c2]
      matched_kp = kp2[idx]
      matched_kp2.append(matched_kp)
      c2 += 1

    matched_kp1 = np.array(matched_kp1)
    matched_kp2 = np.array(matched_kp2)   
    matched_keypoints = []
    for i in range(len(matched_kp1)):
      kp1_item = matched_kp1[i]
      kp2_item = matched_kp2[i]
      combined_kp = np.hstack((kp2_item, kp1_item))
      matched_keypoints.append(combined_kp)
    matched_keypoints = np.array(matched_keypoints)

    
    return matched_keypoints




img1 = imread('/content/drive/MyDrive/pset3/data/left.jpg')
img2 = imread('/content/drive/MyDrive/pset3/data/right.jpg')

data = get_best_matches(img1, img2, 300)
fig, ax = plt.subplots(figsize=(20,10))
plot_inlier_matches(ax, img1, img2, data)
fig.savefig('sift_match.pdf', bbox_inches='tight')

"""**b) Homography Estimation and RANSAC .** Implement RANSAC to estimate a homography mapping one image onto the other. Describe the various implementation details, and report all the hyperparameters, the number of inliers, and the average residual for the inliers (mean squared distance between the point coordinates in one image and the transformed coordinates of the matching point in the other image). Also, display the locations of inlier matches in both images.
        
  **Hints:** For RANSAC, a very simple implementation is sufficient. Use four matches to initialize the homography in each iteration. You should output a single transformation that gets the most inliers in the course of all the iterations. For the various RANSAC parameters (number of iterations, inlier threshold), play around with a few reasonable values and pick the ones that work best. Homography fitting calls for homogeneous least squares. The solution to the homogeneous least squares system $AX = 0$ is obtained from the SVD of $A$ by the singular vector corresponding to the smallest singular value. In Python, `U, S, V = numpy.linalg.svd(A)` performs the singular value decomposition (SVD). This function decomposes A into $U, S, V$ such that $A = USV$ (and not $USV^T$) and `V[−1,:]` gives the right singular vector corresponding to the smallest singular value. *Your implementation should not use any opencv functions.*
"""

def ransac(data, max_iters=10000, min_inliers=10):
    """
    Write your ransac code to find the best model, inliers, and residuals
    
    :param data: pixel coordinates of matched keypoints.
    :param max_iters: number of maximum iterations
    :param min_inliers: number of minimum inliers in RANSAC
    :param threshold: threshold for inlier determination
    
    return: (homography matrix, number of inliers) -> (3x3 numpy array, int)
    """
    H_out = None
    best_perf = 0
    m=data
    itr = 0

    while itr < max_iters:
          global inp, out
          ind=len(data)
          inp, out = np.split(data[np.random.choice(ind,6,replace=False), :], [2], axis=1)
          H = compute_homography(m)
          r=1
          num,deno = np.matmul(H, np.concatenate((data[:, :2].T, np.ones((r, data.shape[0]))), axis=0))[:2, :],np.matmul(H, np.concatenate((data[:, :2].T, np.ones((r, data.shape[0]))), axis=0))[2, :]
          new_inpt=(num / deno).T - data[:, 2:]
          res = np.linalg.norm(new_inpt, axis=1)      
          inlier_count = 0
          threshold = 2
          for r in res:
            if r < threshold:
              inlier_count += 1
          il = inlier_count
          if il > best_perf:
            best_perf, H_out = il, H
          itr += 1
    return H_out, best_perf




def compute_homography(matches):
    """
    Write your code to compute homography according to the matches
    
    :param src: source pixel coordinates of matched keypoints.
    :param dst: destination pixel coordinates of matched keypoints.
    
    return: homography matrix, which is a 3x3 numpy array
    """
    K = []
    i = 0
    while i < inp.shape[0]:
        K = [elem for i in range(inp.shape[0]) for elem in [[inp[i, 0], inp[i, 1], 
        1, 0, 0, 0, -out[i, 0] * inp[i, 0], -out[i, 0] * inp[i, 1], -out[i, 0]], [0, 0, 0, inp[i, 0], inp[i, 1], 1, -out[i, 1] * inp[i, 0], -out[i, 1] * inp[i, 1], -out[i, 1]]]]
        i += 1
    H = (np.linalg.svd(np.array(K))[2][-1, :].reshape((3, 3)) / np.linalg.svd(np.array(K))[2][-1, -1])


    return H


np.random.seed(1237)
# Report the number of inliers in your matching
H, max_inliers = ransac(data)
print("Inliers:", max_inliers)

"""**c) Image Warping .** Warped one image onto the other using the estimated transformation. You can use opencv functions for this purpose."""

import cv2

def warp_images(img1, img2, H):
    """
    Write your code to stitch images together according to the homography
    
    :param img1: left image
    :param img2: right image
    :param H: homography matrix
    
    return: stitched image, should be a HxWx3 array.
    """

    for i, dim in enumerate(img1.shape):
      if i == 0:
        len = dim
      elif i == 1:
        bret = dim
      else:
        break

    img_warped = cv2.warpPerspective(img2, cv2.getPerspectiveTransform(np.float32([[0, 0], [0, len], 
    [bret, len], [bret, 0]]), cv2.perspectiveTransform(np.float32([[0, 0], [0, len], [bret, len], [bret, 0]])[:, np.newaxis], H).reshape(-1, 2)), 
    (int(bret*2), int(len*1)), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 0, 0)); img_warped[:len, :bret] = cv2.addWeighted(src1=img1, alpha=1.0, src2=img_warped[:len, :bret], beta=0.0, gamma=0.0)

    return img_warped



# Display the stitching results
img_warped = warp_images(img1, img2, H)
display(Image.fromarray(img_warped))

"""**d) Image Warping  .** Warped one image onto the other using the estimated transformation *without opencv functions*. Create a new image big enough to hold the panorama and composite the two images into it. You can composite by averaging the pixel values where the two images overlap, or by using the pixel values from one of the images. Your result should look similar to the sample output. You should create **color panoramas** by applying  the same compositing step to each of the color channels separately (for estimating the transformation, it is sufficient to use grayscale images). You may find `ProjectiveTransform` and warp functions in `skimage.transform` useful."""

import skimage.transform
import numpy as np

def warp_images_noopencv(img1, img2, H):
    """
    Write your code to stitch images together according to the homography
    
    :param img1: left image
    :param img2: right image
    :param H: homography matrix
    
    return: stitched image, should be a HxWx3 array.

    """
 
    heights = []
    widths = []

    for img in [img1, img2]:
      h, w = img.shape[:2]
      heights.append(h)
      widths.append(w)

    l1, b1 = heights[0], widths[0]

    cn1 = np.zeros((5, 2))
    for i in range(5):
      if i == 0:
        cn1[i] = [0, 0]
      elif i == 1:
        cn1[i] = [0, l1]
      elif i == 2:
        cn1[i] = [b1, l1]
      elif i == 3:
        cn1[i] = [b1, 0]
      elif i == 4:
        cn1[i] = [0, 0]
    ones = np.ones((5, 1))
    cn2homo = np.zeros((5, 3))
    for i, corner in enumerate(np.hstack((cn1, ones))):
      cn2homo[i, :] = np.dot(H, corner.reshape(3, 1)).T

    cns_x, cns_y = (cn2homo[:-1, :2] / cn2homo[:-1, 2:]).T
    x_min, x_max, y_min, y_max = map(int, [min(cns_x.min(), 0), max(cns_x.max(), b1), min(cns_y.min(), 0), max(cns_y.max(), l1)])

    while x_min < 0:
      x_min += 1
      x_max += 1

    while x_max > b1:
      x_min -= 1
      x_max -= 1

    while y_min < 0:
      y_min += 1
      y_max += 1

    while y_max > l1:
      y_min -= 1
      y_max -= 1
    output_shape = (y_max - y_min, x_max - x_min)
    output_rows = []
    for i in range(output_shape[0]):
      row = np.zeros((output_shape[1], 3), dtype=np.uint8)
      output_rows.append(row)  
    img_warped = skimage.transform.warp(img2, np.linalg.inv(np.array([[1-1+1**1,0,-x_min+1],[0,1**1,-y_min+2],[0,0,1]])) @ H, output_shape=output_shape, order=0, mode='constant', cval=0, clip=True, preserve_range=True)
    
    return img_warped



# display and report the stitching results
img_warped_nocv = warp_images_noopencv(img1, img2, H)
display(Image.fromarray(img_warped))

