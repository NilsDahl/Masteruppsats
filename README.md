Building an affine term structure model of yields for our Master Thesis in Economics at SSE

yields.py – constructs zero coupon yields for Swedish inflation linked government bonds and nominal bonds. 

pca.py – takes the nominal zero coupon yields and summarizes the entire nominal yield curve using three principal components that match the shape in Abraham’s specification.

liquidity.py – retrieves turnover data from the Riksbank API database, removes primary market transactions and irrelevant contracts, and constructs a measure of relative illiquidity between real and nominal government bonds over time. 

regression_real.py – purges the real yields from nominal PCA factors and liquidity through regression, and then applies PCA to the residuals in order to isolate pure real yield factors. After this step, we theoretically have all the inputs required for the model. The final step writes the three nominal PCs, the liquidity variable, and the two real PCs to an Excel file that can be used later in the modeling stage.
