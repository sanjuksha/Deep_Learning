 vid=VideoReader('IMG_0757.MOV');
 
 numFrames = vid.NumberOfFrames;
 n=numFrames;
 for i = 1:n
    frames = read(vid,i);
    B=imrotate(frames,270);
    imwrite(B,['sanjuksha' int2str(i), '.jpg']);
    im(i)=image(B);
 end