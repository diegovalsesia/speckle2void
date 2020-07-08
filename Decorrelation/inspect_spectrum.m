function inspect_spectrum(input_file)


complex_SAR = load(input_file);
fields=fieldnames(complex_SAR);
inphase = getfield(complex_SAR,fields{2});
inquad = getfield(complex_SAR,fields{1});
img_complex = double(inphase) + 1j*double(inquad);

[r,c]=size(img_complex);

fC = fft2(img_complex);
S = real(fC.*conj(fC));
temp1=sqrt(mean(fftshift(S), 1));
temp1 = temp1/max(temp1);
x1 = -1:2/(r-1):1;
figure, plot(x1, temp1, 'o')
temp1=sqrt(mean(fftshift(S), 2));
temp1 = temp1/max(temp1);
x1 = -1:2/(c-1):1;
figure, plot(x1, temp1, 'o')
    
end