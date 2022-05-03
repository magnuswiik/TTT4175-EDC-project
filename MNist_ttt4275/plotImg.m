function img = plotImg(flat_img, prediction, label, correct)
img = zeros(28,28);
img(:) = flat_img(:);
figure(1)
image(rot90(flip(img),3));
text = "|" + correct + "|" + " Prediction: " + num2str(prediction) + " Label: " + num2str(label);
title(text);
end

