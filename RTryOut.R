library(EMD)
library(imager)
path <- 'C:/Users/tomda/Desktop/Projects/מחקר HHT-SR/Repository/Hilbert-Huang-Transform-Super-Resolution/DATA/Ross.jpg'
path <- system.file(path, package = 'imager')
img <- load.image(path)
