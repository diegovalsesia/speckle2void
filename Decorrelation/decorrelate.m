function decorrelate(input_file,output_file,f_x,f_y,m_x,m_y,cf)


complex_SAR = load(input_file);
fields=fieldnames(complex_SAR);
inphase = getfield(complex_SAR,fields{2});
inquad = getfield(complex_SAR,fields{1});
img_complex = double(inphase) + 1j*double(inquad);

[r,c]=size(img_complex);
intensity_img = abs(img_complex).^2;
median(intensity_img(:))
threshold = cf*median(intensity_img(:));

index_nonpoints = intensity_img<threshold;
index_points = intensity_img>=threshold;

n_nonpoints = sum(index_nonpoints(:));
n_points = sum(index_points(:));
var = sum(intensity_img(index_nonpoints)/n_nonpoints);

%replace the point targets with complex values drawn from a complex
%circular symmetric Gaussian distribution
new_points=sqrt(var/2)*(randn(1,n_points)+1j*randn(1,n_points));

img_complex_new = img_complex;
img_complex_new(index_points)=new_points;


[cout W] = equalizeIQ_my_polynomio(img_complex_new,m_x,m_y,f_x,f_y);

%plot the frequency spectrum of the decorrelated complex SAR image
fC = fft2(cout);
S = real(fC.*conj(fC));
temp1=sqrt(mean(fftshift(S), 1));
temp1 = temp1/max(temp1);
x1 = -1:2/(r-1):1;
figure, plot(x1, temp1, 'o')
temp1=sqrt(mean(fftshift(S), 2));
temp1 = temp1/max(temp1);
x1 = -1:2/(c-1):1;
figure, plot(x1, temp1, 'o')

%replace back the point targets
cout(index_points) = img_complex(index_points);


save(output_file,'cout');

    
end