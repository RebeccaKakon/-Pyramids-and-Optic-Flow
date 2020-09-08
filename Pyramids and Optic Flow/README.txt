1. Lucas Kanade optical flow
a function which takes an image and returns the optical flow by using the LK algorithm.
Given two images, returns the Translation from im1 to im2


2.Gaussian and Laplacian Pyramids
-gaussianPyr this function Creates a Gaussian Pyramid . 
with the input of Original image and the Pyramid dept

-gaussExpand this function expands a Gaussian pyramid level one step up . 
the function input Pyramid image at a certain level 
and The kernel to use in expanding. reurn the expanded level

-laplaceianReduce Creates a Laplacian pyramid. 
function input Original image and Pyramid depth
and return a Laplacian Pyramid (list of images)

-laplaceianExpand Resotrs the original image from a laplacian pyramid input Laplacian Pyramid
and returnthe Original image

-pyrBlendBlends two images using PyramidBlend method
input Image 1 and Image 2, Blend mask and a Pyramid depth 
return (Naive blend, Blended Image)
the naiv is the Formula we will do foe each level but now only for the imge given 
