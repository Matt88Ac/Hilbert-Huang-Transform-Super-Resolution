for k =str2num('02000'): str2num('02100')
temp1 = num2str(k)
temp2 = 'DATA/'
temp3 = '.jpg'
temp4 = '0'
i= imread(strcat(temp2,temp4,temp1, temp3))


i=rgb2gray(i)

i= i

a=bemd(i)

[m,n,k] = size(a)
end
%s = a(:,:,1) +a(:,:,2)+a(:,:,3)+a(:,:,4)
%b = cast(s,'uint8')
%imshow(b)