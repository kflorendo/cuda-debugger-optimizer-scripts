# CMU 15-418/618, Spring 2023

# Assignment 2

This is the starter code for Assignment 2 of CMU class 15-418/618, Spring 2023

Please review the course's policy on [academic
integrity](http://www.cs.cmu.edu/~418/academicintegrity.html),
regarding your obligation to keep your own solutions private from now
until eternity.

## Conceptual Idea

parallelize over pixels because if we parallelize over circles, it's hard to know what order the circles should go

parallel_for each pixel:
    overlapping_circles = []
    parallel_for each 0 < index < numCircles:
        if circles[index] overlaps pixel:
            add index to overlapping_circles
    for each index in overlapping_circles (sorted):
        calculate new color of pixel after overlaying circle on top


use exclusive scan and indicator array for circles that overlap
diff ways to do this:
- identify circles and color pixels in different kernels
- identify circles and color pixels in same kernel (might need alternating stuff?)


