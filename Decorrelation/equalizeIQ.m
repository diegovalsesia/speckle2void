function [CE,W] = equalizeIQ(C, mx, my, rx, ry)
%
% C: single look complex data
% mx, my: frequency shift in normalized frequencies (pi*m)
% rx, ry: only frequencies in [-pi(1-r), pi(1-r)] are equalized
%
% CE: equalized complex data
% W: frequency mask of whitening filter

[r,c] = size(C);
M = exp(-j*pi*(0:r-1)*my).' * exp(-j*pi*(0:c-1)*mx);
max(M(:));
min(M(:));
C2 = M .* C;
% C2 = C;

fC = fft2(C2);


S = real(fC.*conj(fC));
%figure, imshow(fftshift(sqrt(S)), [])
%figure, imshow(sqrt(S), [])
R = ifft2(S);
rho_x = abs(R(1,2)/R(1,1))
rho_y = abs(R(2,1)/R(1,1))
%figure, imshow(fftshift(abs(R)), [])

fun = @(p,xdata) p(4)*(p(1) - (1-p(1))*cos(pi*(xdata-p(3))/p(2)));

r1 = round(c*rx);
x1 = 0:c-1;
y1 = sqrt(mean(fftshift(S), 1));
clipping = prctile(y1,99);
y1(y1>clipping)=clipping;
%max(y1);
y1 = y1/max(y1);


p = polyfit(x1(1+r1:end-r1+1), y1(1+r1:end-r1+1), 70);
yi1 = polyval(p, x1);
figure, plot(x1, y1,'*', x1(1+r1:end-r1+1), yi1(1+r1:end-r1+1),'o')

%options = optimoptions('lsqcurvefit','StepTolerance', 1.0000e-08,'FunctionTolerance',1e-15,'OptimalityTolerance',1e-12);
%lb = [];
%ub = [];
%p1 = lsqcurvefit(fun, [0.5 r/2 0 1], x1(1+r1:end-r1+1), y1(1+r1:end-r1+1),lb,ub,options)

r2 = round(r*ry);
x2 = (0:r-1)';
y2 = sqrt(mean(fftshift(S), 2));
clipping = prctile(y2,99);
y2(y2>clipping)=clipping;
max(y2);
y2 = y2/max(y2);



p = polyfit(x2(1+r2:end-r2+1), y2(1+r2:end-r2+1), 70);
yi2 = polyval(p, x2);
figure, plot(x2, y2,'*', x2(1+r2:end-r2+1), yi2(1+r2:end-r2+1),'o')

%p2 = lsqcurvefit(fun, [0.5 r/2 0 1], x2(1+r2:end-r2+1), y2(1+r2:end-r2+1), lb,ub,options)

G = ((yi2) * yi1);
mask = zeros(r,c);
mask(1+r2:end-r2+1,1+r1:end-r1+1) = 1;
G = G .* mask;
G = G/norm(G(:));
W = 1 ./ G / sqrt(sum(mask(:)));
W(~mask) = 1;
sum((G(:).*W(:)).^2)
% W(G <= 0) = 1/0.1;
% w2 = hamming(c);
% w1 = hamming(r);
% sig2 = c*2;
% sig1 = r*2;
% w2 = exp(-((-(c-1)/2:(c-1)/2)/sig2).^2/2).';
% w1 = exp(-((-(r-1)/2:(r-1)/2)/sig1).^2/2).';
% W = W .* (w2 * w1.');

W = fftshift(W);
%figure, imshow(W, [])
% P = sum(S(:))
% PE = sum(S(:).*W(:).^2)
% W = W * sqrt(P/PE);

%filter high frequency
h=zeros(r,c);
r1 = round(c*rx);
r2 = round(r*ry);
h(1+r2:end-r2,1+r1:end-r1)=1;
h=fftshift(h);
%figure, imshow(fftshift(abs(fC)), [])
fC=fC.*h;

fCE = fC .* W;
%figure, imshow(fftshift(abs(fCE)), [])
CE = ifft2(fCE);

RE = ifft2(S.*W.^2);
rho_xe = abs(RE(1,2)/RE(1,1))
rho_ye = abs(RE(2,1)/RE(1,1))
%figure, imshow(fftshift(abs(RE)), [])

return
