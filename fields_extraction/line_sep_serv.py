import numpy as np
import cv2

"""
This method is used to extract the lines from an address block
@param block : the address block

@return date_eng  : the image of the english date of expiry
@return date_thai : the image of the thai date of expiry
@return valid     : whether the lines have been extracted nicely
@return drawing   : debuging purpose, each line drawn with color
"""
def apply_line_extraction_on_block(block) :


    """
    This method is used to check whether the line detection is valid or not
    @param arrays     : the each array is contains successive indices of text-detected lines 
    @param ideal_dist : the ideal distance between two successive lines of text
    @param threshold  : the smaller the stricter we are with validity
    """
    def check_valid(arrays, ideal_dist, threshold) :
        for i, arr in enumerate(arrays) :
            
            if i != 0 :

                #lines are too (or not enough) distant from each other
                if abs(((np.median(np.array(arr)) - np.median(np.array(arrays[i - 1]))) / ideal_dist) - 1) > threshold :
                    return False
        return True


    """
    compute sequences of zero and non-zero values in an array
    @param to black : the sequence of values

    @return arrays_zeros     : a list of list of successive zero-value indices
    @return arrays_non_zeros : a list of list of successive non-zero-value indices

    example :
    to_black         = [0, 0, 0, 1, 2, 4, 2, 0, 0, 8]
    arrays_zeros     = [[0, 1, 2], [7, 8]]
    arrays_non_zeros = [[3, 4, 5, 6], [9]]
    """
    def compute_sequences(to_black) :
        #compute starts and ends of 0s subsequences
        last_zero = to_black[0] != 0

        arrays_zeros = [[]]
        arrays_non_zeros = [[]]

        cur = []
        for i, elm in enumerate(to_black) :

            #end of the array
            if i == len(to_black) - 1:
                if last_zero and elm == 0:
                    cur.append(i)
                    arrays_zeros.append(cur)

                elif not last_zero and elm != 0 :
                    cur.append(i)
                    arrays_non_zeros.append(cur)

                elif last_zero and elm != 0 :
                    arrays_zeros.append(cur)
                    arrays_non_zeros.append([i])

                else :
                    arrays_non_zeros.append(cur)
                    arrays_zeros.append([i])

            elif elm == 0 and not last_zero :
                arrays_non_zeros.append(cur)
                cur = []
                cur.append(i)
                last_zero = True

            elif (elm == 0 and last_zero) or (elm !=0 and not last_zero):
                cur.append(i)

            elif elm !=0 and last_zero :
                arrays_zeros.append(cur)
                cur = []
                cur.append(i)
                last_zero = False
        return arrays_zeros, arrays_non_zeros
 
    
    
    
    """
    This function is used to filter line of text containing the dates
    @param arrs : the output of the compute_sequences function (arrays_non_zero)

    @return the array containing the indices of rows containing thai    date
    @return the array containing the indices of rows containing english date
    """
    def filter_lines(arrs) :

        #keep only beginning indices
        begins = [arr[0] for arr in arrs]

        #sort them
        sorted_begins = np.argsort(begins)

        #return first and thirs arrays sorted by height
        return arrs[sorted_begins[0]], arrs[sorted_begins[2]]

    
    
    
    
    
    
    img = block

    #whether we found 4 peaks or more
    found = False

    #what we substract to have peaks emerge
    acc = 0

    scale = img.shape[0] / 86

    ideal_width = 20 * scale
    ideal_dist = img.shape[0]/2

    threshold = 0.4
    org = np.copy(img)

    while not found :

        #binarize image
        _, img = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY_INV)
        result = np.copy(img)

        #compute histogram
        hist = np.sum(img, axis = 1) / 255
        thresh = np.mean(hist)

        #substract mean sum of pixel values per row and keep those > 0
        keep = ((hist - thresh) - acc)

        # for each row, 0 iff it isn't a line of text
        to_black = np.copy(keep)
        to_black[keep < 0] = 0

        #compute sequences
        arrays_zeros, arrays_non_zeros = compute_sequences(to_black)
        arrays_zeros = np.array([i for i in arrays_zeros if len(i) > 0])
        arrays_non_zeros = np.array([i for i in arrays_non_zeros if len(i) > 0])

        #non interesting lines are set to black
        for arr in arrays_zeros :
            result[arr] = 0

        nb_arr = len(arrays_non_zeros)


        
        found = nb_arr >= 4
        if not found :

            #substract bigger value in case we do not have enough peaks
            acc +=img.shape[0]/50
            if acc + thresh > 255 :
                return None


    #We found exactly the good number of peaks
    if nb_arr == 4 :


        #english and thai dates indices
        to_keep = filter_lines(arrays_non_zeros)
        valid = check_valid(to_keep, ideal_dist, threshold)

        drawing = np.copy(img)
        drawing = cv2.cvtColor(drawing,cv2.COLOR_GRAY2BGR)
        images = []

        for i,line in enumerate(to_keep) :

            #what we need to pad to have line of width ideal_width
            to_pad = max(ideal_width - len(line), 0)

            #new borders after padding
            l = list(range(max(int(line[0] - to_pad/2), 0), line[0]))
            r = list(range(line[-1], min(int(line[-1] + to_pad/2), img.shape[0])))


            #row indices of interesting line
            line = l + list(line) + r
            line = np.array(line)

            #draw interesting characters in green
            to_draw = drawing[line]
            to_draw[np.where((to_draw == [255,255,255]).all(axis = 2))] = [0,255,0]
            drawing[line] = to_draw
            sub = org[line]
            images.append(sub)

        date_thai = images[0]
        date_eng = images[1]
        return date_eng, date_thai, valid, drawing


    else :

        #score of a peak is (max pixel intensity * width)
        scores = []
        for arr in arrays_non_zeros :
            max_arr = np.max(np.sum(result[arr], axis = 1) / 255)
            len_arr = len(arr)
            score_arr = max_arr*len_arr
            scores.append(score_arr)

        #keep 4 peaks with best scores
        to_keep = np.argsort(scores)[::-1][:4]

        #keep only rows with date
        arrs = [arrays_non_zeros[i] for i in to_keep]
        to_keep = filter_lines(arrs)
        valid = check_valid(to_keep, ideal_dist, threshold)

        drawing = np.copy(img)
        drawing = cv2.cvtColor(drawing,cv2.COLOR_GRAY2BGR)
        images = []
        for i,line in enumerate(to_keep) :

            #what we need to pad to have line of width ideal_width
            to_pad = max(ideal_width - len(line), 0)

            #new borders after padding
            l = list(range(max(int(line[0] - to_pad/2), 0), line[0]))
            r = list(range(line[-1], min(int(line[-1] + to_pad/2), img.shape[0])))

            line = l + list(line) + r

            line = np.array(line)

            to_draw = drawing[line]
            to_draw[np.where((to_draw == [255,255,255]).all(axis = 2))] = [0,255,0]
            drawing[line] = to_draw
            sub = org[line]
            images.append(sub)

        date_thai = images[0]
        date_eng = images[1]
        return date_eng, date_thai, valid, drawing
