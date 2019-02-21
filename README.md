# Project-NST
## Description: 
A practice demo project using neural style transfer to create 'art' pictures.
## Basic Information
The pretrained model used is VGG-19. Unfortunately the .mat file of the model is too large to store in Github.

To download the model you may find various sources e.g. http://www.vlfeat.org/matconvnet/models/beta16/.

The content image is a photo of Dalian at dusk and the style image is the famous painting *The Starry Night* by Vincent van Gogh. Both images are contained in folder *images*.

The model outputs one generated image every 20 iterations and all the output images are stored in the folder *output* where you can clearly see the style 'grows' on the content image. You will also notice a decreasing total cost and style cost while running the program.

## Fun Fact
I have also tried NST using my own selfie between the self-portrait of van Gogh, Rembrandt and Renoir. To save you from the terror of the output image, I will not upload the results. 

My guess is, the NST method tends to applied the extracted features to the whole image, in which case the landscape paintings/photos are fine. However, when it comes to human faces, the situation becomes complicated possibly because the face part does not share the same features as other parts of the images e.g. the background. To deploy NST on selfies it may require facial recognition techniques and I will be digging into that in the near future. If you have any great idea, leave a comment!
