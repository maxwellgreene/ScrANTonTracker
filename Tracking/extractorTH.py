# class extractorTH:
#    def __init__(self):
#        self.num_ants = None
#        self.antsWithCont = []
#        self.antsWithoutCont = []
#        self.thresh = 145
#        self.thresh_shift = 0
#        self.valid_contours = []
#        self.min = 0
#        self.max = 255
#
#    def set_ant_num(self, a_num_ants):
#        """ Used to manualy set the nuymber of ants.
#        Latter we should add in the abilty to do this manualy on the first frame
#        if it has not been set manualy"""
#        # if isinstance(x, int):
#        self.num_ants = a_num_ants
#
#    def get_num_ants(self):
#        num_ants = int(input("How many ants are there? \n"))
#        self.set_ant_num(num_ants)
#
#    def get_len_width(self, a_box):
#        p1 = a_box[0]
#
#        distances = []
#        for p in a_box[1:]:
#            d = np.sqrt((p[0] - p1[0]) ** 2 + (p[1] - p1[1]) ** 2)
#            distances.append(d)
#
#        distances.sort()
#        return distances[:-1]
#
#    def get_len_width_theta(self, a_box):
#
#        p1 = a_box[0]
#
#        distances = []
#        thetas = []
#
#        for p in a_box[1:]:
#            dx = p[0] - p1[0]
#            dy = p[1] - p1[1]
#            if dx == 0:
#               thetas.append(90.0)
#            else:
#                thetas.append(np.arctan(dy / dx) * 180 / 3.14159)
#
#            distances.append(np.sqrt(dx ** 2 + dy ** 2))
#
#        out = sorted(zip(distances, thetas))
#        distances, thetas = zip(*out)
#        return distances[0], distances[1], thetas[1]
#
#    def get_corners(self, length, width, theta, center_x, center_y, scale):
#        xt = scale * length / 2 * np.cos(theta * 3.14159 / 180)
#        yt = scale * length / 2 * np.sin(theta * 3.14159 / 180)
#
#        xs = scale * width / 2 * np.cos((theta + 90) * 3.14159 / 180)
#        ys = scale * width / 2 * np.sin((theta + 90) * 3.14159 / 180)
#
#        p1 = [int(center_x + xt + xs), int(center_y + yt + ys)]
#        p2 = [int(center_x + xt - xs), int(center_y + yt - ys)]
#        p3 = [int(center_x - xt - xs), int(center_y - yt - ys)]
#        p4 = [int(center_x - xt + xs), int(center_y - yt + ys)]
#
#        return np.array([p1, p2, p3, p4])
#
#    def make_markers(self, ants, a_frame):
#        x, y = a_frame.shape  # I did
#        markers = np.zeros((x, y), np.uint8)
#
#        polys = []
#        for ant in ants:
#            tempMask = np.zeros((x, y), np.uint8)
#            tempMask[a_frame > 0] = 255
#
#            xp, Pp = ant.predict(ant.xk, ant.Pk)
#            length = xp[3]
#            width = xp[4]
#            theta = xp[2]
#            center_x = xp[0]
#            center_y = xp[1]
#            box = self.get_corners(length, width, theta, center_x, center_y, .25)
#            poly = Polygon(box)
#
#            antMask = mask_for_polygons([poly], a_frame.shape)
#            tempMask[antMask == 0] = 0
#
#            if np.amax(tempMask) > 0:
#                markers[tempMask > 0] = 255
#            else:
#                markers[antMask > 0] = 255
#
#            polys.append(poly)
#
#
#        # make the unknown regions
#        unknown = np.zeros((x, y), np.uint8)
#        unknown[a_frame > 20] = 255
#        mask = mask_for_polygons(polys, a_frame.shape)
#        unknown[mask > 0] = 255
#        # mask out the ant centers
#        unknown[markers > 0] = 0
#
#        # make the final marker image
#        # Marker labelling
#        ret, markers = cv2.connectedComponents(markers)
#        # Add one to all labels so that sure background is not 0, but 1
#        markers = markers + 1
#        # Now, mark the region of unknown with zero
#        markers[unknown == 255] = 0
#
#        return markers
#
#
#
#    def draw_boarders(self, ants, a_frame):
#        for ant in ants:
#            xp, Pp = ant.predict(ant.xk, ant.Pk)
#            length = xp[3]
#            width = xp[4]
#            theta = xp[2]
#            center_x = xp[0]
#            center_y = xp[1]
#            box = self.get_corners(length, width, theta, center_x, center_y, 1.)
#
#            cv2.drawContours(a_frame, [box], 0, (0, 0, 0), 2)
#
#        return a_frame
#
#    def makeBlobs(self, a_img):
#        gray = a_img[:,:,0] #cv2.cvtColor(a_img, cv2.COLOR_BGR2GRAY)
#        #clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4,4))
#        print_im(gray)
#        gray = (255/(self.M-self.m))*(gray-self.m)
#        gray[gray>255] = 255
#        print('rescaled', self.m, self.M)
#        print(np.min(gray), np.max(gray))
#        g = gray.astype("uint8")
#        gray = g
#        print_im(gray)
#
#        # blurred = cv2.GaussianBlur(gray, (11, 11), 0)
#
#        ret, thresh = cv2.threshold(gray, self.thresh, 255, cv2.THRESH_BINARY_INV)
#
#        """
#        thresh2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,55,50)
#        thresh3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,55,50)
#        print_im(thresh)
#        print_im(thresh2)
#        print_im(thresh3)
#        """
#        print("thresh")
#        print_im(thresh)
#        closedIM = self.close(thresh  , 5)
#        """
#
#        # Perform the operation
#        output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
#        # Get the results
#        # The first cell is the number of labels
#        num_labels = output[0]
#        # The second cell is the label matrix
#        labels = output[1]
#        # The third cell is the stat matrix
#        stats = output[2]
#        # The fourth cell is the centroid matrix
#        centroids = output[3]
#        for ant in ants:
#            tempMask = np.zeros((x, y), np.uint8)
#            tempMask[a_frame > 0] = 255
#
#            xp, Pp = ant.predict(ant.xk, ant.Pk)
#            length = xp[3]
#            width = xp[4]
#            theta = xp[2]
#            center_x = xp[0]
#            center_y = xp[1]
#            box = self.get_corners(length, width, theta, center_x, center_y, .25)
#            poly = Polygon(box)
#
#        """
#        #print_im(closedIM)
#        return closedIM
#
#
#    def filterContours(self, a_contours):
#        areas = []
#        lens = []
#
#        # get info on all the connected components
#        for i, cnt in enumerate(a_contours):
#            area = cv2.contourArea(cnt)
#
#            rect = cv2.minAreaRect(cnt)
#            box = cv2.boxPoints(rect)
#            box = np.int0(box)
#            w, l, theta = self.get_len_width_theta(box)
#
#            lens.append(l)
#            areas.append(area)
#
#        if len(areas) ==0:
#            return []
#        median_len = statistics.median(lens)
#        median_area = statistics.median(areas)
#
#        contours_out = []
#
#        for a, l, cnt in zip(areas, lens, a_contours):
#            if 100 * a / median_area > 60 and 100 * l / median_len > 60:  # apply median filter
#                contours_out.append(cnt)
#
#        return contours_out
#
#    def resetForRun(self):
#        self.valid_contours = []
#        self.antsWithCont = []
#        self.antsWithoutCont = []
#
#    def findAnts(self, a_frame, a_ants, a_frame_num):
#        allFound = False
#
#        #if a_frame_num == 0:
#            #print_im(a_frame[:,:,0])
#            #print_im(a_frame[:,:,1])
#            #print_im(a_frame[:,:,2])
#
#        # make sure we know how many ants there should be.
#        if self.num_ants is None:
#            self.get_num_ants()
#
#        # reset the containers
#        self.resetForRun()
#
#        count = 0
#        ants2find = set(range(len(a_ants)))
#
#        while not allFound:
#            count += 1
#            #print("the threshold is ", self.thresh)
#            # Get the connected components of the threshold image
#            closedIM = self.makeBlobs(a_frame)
#            print_im(closedIM)
#            # get the contours of the threhold blobs
#            contours, hierarchy = cv2.findContours(closedIM, 1, 2)[-2:]
#            #cv2.connectedComponentsWithStats(closedIM, 8, cv2.CV_32S)
#
#            #if len(contours) > self.num_ants:
#            contours = self.filterContours(contours)
#            print("we found ", len(contours), " ants")
#
#            # check how we did and adjust things if need
#            if a_frame_num > 0:
#                #if len(contours) == self.num_ants:  # everything is good on the first frame and we can proceed.
#                self.valid_contours = contours
#                allFound = True
#                continue
#
#                markers = self.make_markers(a_ants, closedIM)
#                # plt.figure(figsize=(12, 4), dpi= 100, facecolor='w', edgecolor='k')
#                # plt.imshow(markers)
#                # plt.show()
#                markers = cv2.watershed(a_frame, markers)
#                #mset = np.unique(markers)
#                #print(len(mset), " and the set is ", mset)
#
#                #plt.figure(figsize=(12, 4), dpi= 100, facecolor='w', edgecolor='k')
#                #plt.imshow(markers)
#                #plt.show()
#                allFound = True
#
#                for i in np.arange(2, np.amax(markers) + 1):
#                    #print(i)
#                    tmask = np.zeros(markers.shape, np.uint8)
#                    tmask[markers == i] = 255;
#
#                    conts, hierarchy = cv2.findContours(tmask, 1, 2)[-2:]
#                    self.valid_contours.append(conts[0])
#                #print("len valid_conts", len(self.valid_contours))
#            # if we are a on the first frame make sure we have the write number of ants
#            if a_frame_num == 0 and len(contours) < self.num_ants:  # threshold is too low
#                self.thresh += 1
#                print("changeTheshUp")
#            elif a_frame_num == 0 and len(contours) > self.num_ants:  # threshold is too high
#                print("changeThesh down")
#                self.thresh -= 1
#            elif a_frame_num == 0:  # everything is good on the first frame and we can proceed.
#                self.valid_contours = contours
#                allFound = True
#
#            # if this fails then the frame is bad and we need to proceed without it
#            if self.thresh <= 30 or self.thresh >= 230 or count >= 200:
#                self.thresh = 90
#                self.thresh_shift = 0
#                return [], False
#
#        # print(len(contours))
#        contours = self.valid_contours
#
#        areas = []
#        boxes = []
#        centers = []
#        lens = []
#        widths = []
#        thetas = []
#        meas_vecs = []
#
#        for i, cnt in enumerate(contours):
#            area = cv2.contourArea(cnt)
#
#            rect = cv2.minAreaRect(cnt)
#            box = cv2.boxPoints(rect)
#            box = np.int0(box)
#            w, l, theta = self.get_len_width_theta(box)
#
#            # compute the center of the contour
#            M = cv2.moments(cnt)
#            if M["m00"] < .0001: continue
#            cx = int(M["m10"] / M["m00"])
#            cy = int(M["m01"] / M["m00"])
#            centers.append([cx, cy])
#
#            lens.append(l)
#            widths.append(w)
#            thetas.append(theta)
#            areas.append(area)
#            boxes.append(box)
#
#            # print('out', w,l,theta)
#            meas_vecs.append([cx, cy, theta, l, w])
#            cv2.drawContours(a_frame, [box], 0, (0, 0, 255), 2)
#            # print_im(a_frame)
#
#
#
#        # print_im(a_frame)
#        # print(len(meas_vecs))
#        if len(meas_vecs) == self.num_ants:
#            # print('good',len(keep_centers))
#            return meas_vecs, True
#        else:
#            # print('bd',len(keep_centers))
#            return meas_vecs, True
#            #return [], False
#
#    def close(self, a_frame, a_size):
#        kernel = np.ones((3, 3), np.uint8)
#        erosion = cv2.erode(a_frame, kernel, iterations=2)
#        #print_im(erosion)
#        kernel = np.ones((a_size, a_size), np.uint8)
#        #dilation = cv2.dilate(erosion, kernel, iterations=3)
#        #print_im(dilation)
#        return erosion
