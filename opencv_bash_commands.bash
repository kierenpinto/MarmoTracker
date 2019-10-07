#bash scripts

opencv_createsamples -img face02.jpg -bg bg.txt -info info/info.lst -pngoutput positive_images -maxxangle 0.5 -axyangle -0.5 -maxzangle 0.5 -num 1950

opencv_createsamples -info positive_images/info.lst -num 1950 -w 20 -h 20 -vec positives.vec

opencv_traincascade -data -data -vec positives.vec -bg bg.txt -numPos 1800 -numNeg 900 -numStages 10 -w 20 -h 20