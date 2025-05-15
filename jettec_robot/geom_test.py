# -*- coding: utf-8 -*-

import pyzed.sl as sl
import cv2
import numpy as np

def create_trapezoidal_mask(shape):
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)

    bottom_left = (int(width * 0.05), height)
    bottom_right = (int(width * 0.95), height)
    top_left = (int(width * 0.4), int(height * 0.55))
    top_right = (int(width * 0.6), int(height * 0.55))

    roi_corners = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, [roi_corners], 255)
    return mask

def perspective_transform(image):
    height, width = image.shape

    src = np.float32([
        [int(width * 0.4), int(height * 0.55)],
        [int(width * 0.6), int(height * 0.55)],
        [int(width * 0.95), height],
        [int(width * 0.05), height]
    ])

    dst = np.float32([
        [int(width * 0.25), 0],
        [int(width * 0.75), 0],
        [int(width * 0.75), height],
        [int(width * 0.25), height]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    birdseye = cv2.warpPerspective(image, M, (width, height))

    return birdseye, Minv

def sliding_window_polyfit(binary_warped, original_img, Minv):
    height, width = binary_warped.shape
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    histogram = np.sum(binary_warped[height//2:,:], axis=0)
    base_x = np.argmax(histogram)

    nwindows = 9
    window_height = int(height / nwindows)
    x_current = base_x
    lane_inds = []

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 60
    minpix = 50

    for window in range(nwindows):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin

        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]

        lane_inds.append(good_inds)

        if len(good_inds) > minpix:
            x_current = int(np.mean(nonzerox[good_inds]))

    lane_inds = np.concatenate(lane_inds)

    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]

    if len(x) > 0 and len(y) > 0:
        fit = np.polyfit(y, x, 2)
        ploty = np.linspace(0, height-1, height)
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]

        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts = np.array([np.transpose(np.vstack([fitx, ploty]))])
        cv2.polylines(color_warp, np.int32([pts]), isClosed=False, color=(0, 0, 255), thickness=5)

        newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
        result = cv2.addWeighted(original_img, 1, newwarp, 0.6, 0)

        return result

    return original_img

def main():
    # --- Initialiser ZED
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Ou HD1080
    init_params.camera_fps = 30

    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Erreur d'ouverture de la caméra ZED :", status)
        exit(1)

    runtime_parameters = sl.RuntimeParameters()

    image_zed = sl.Mat()

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # Récupérer l'image RGB
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            frame = image_zed.get_data()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

            # Traitement Vision
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (7, 7), 0)
            edges = cv2.Canny(blur, 50, 120)

            mask = create_trapezoidal_mask(edges.shape)
            masked_edges = cv2.bitwise_and(edges, mask)

            birdseye, Minv = perspective_transform(masked_edges)

            output = sliding_window_polyfit(birdseye, frame, Minv)

            # Affichage
            cv2.imshow("ZED Birdseye", birdseye)
            cv2.imshow("ZED Detected Curve", output)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
