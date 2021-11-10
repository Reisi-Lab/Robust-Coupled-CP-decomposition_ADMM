function S = soft_thresholding(k,X)
S = (X-k).*((X-k)>0)-(-X-k).*((-X-k)>0);
end